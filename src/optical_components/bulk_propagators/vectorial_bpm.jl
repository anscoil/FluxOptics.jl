struct VectorialBPM{M, T, N, R, P, E, F, K, B} <: AbstractBidirectionalComponent{M}
    trainability::Val{M}
    eps_xyz::N
    eps_ref::R
    dz::T
    n_sub::Int
    kmax2::T
    p_f::P
    medium::E
    factors::F
    kx::K
    ky::K
    buffers::B
    nrm_f::T
end

Functors.@functor VectorialBPM (eps_xyz,)

function VectorialBPM(u::VectorialField, thickness::Real,
                      eps_xyz::AbstractArray{<:Number, 3}, eps_ref::Number;
                      k_max::Real = Inf, n_sub::Integer = 1, trainable::Bool = false)
    T = real(eltype(u.Ex))
    ns = size(u.Ex)[1:2]
    @assert size(eps_xyz)[1:2] == ns
    @assert n_sub >= 1
    dz = T(thickness / (size(eps_xyz, 3) * n_sub))
    eps_buf = similar(u.Ex, isreal(eps_xyz) ? T : Complex{T}, size(eps_xyz))
    copyto!(eps_buf, eps_xyz)
    ε_r = isreal(eps_ref) ? T(real(eps_ref)) : Complex{T}(eps_ref)
    p_f, _ = make_fft_plans(similar(u.Ex), (1, 2); normalize = false)
    medium = VectorialMediumModes(u, ε_r)
    factors = PropagationFactors.(medium.modes, dz, false)
    fx = collect(T, T(2π) .* fftfreq(ns[1], 1 / u.ds[1]))
    fy = collect(T, T(2π) .* fftfreq(ns[2], 1 / u.ds[2]))
    kx, ky = similar(u.Ex, T, ns), similar(u.Ex, T, ns)
    copyto!(kx, repeat(fx, 1, ns[2]))
    copyto!(ky, repeat(permutedims(fy), ns[1], 1))
    buffers = (similar(u.Ex), similar(u.Ex))
    M = trainable ? Trainable : Static
    VectorialBPM(Val(M), eps_buf, ε_r, dz, Int(n_sub), T(k_max)^2, p_f, medium, factors,
                 kx, ky, buffers, T(1 / prod(ns)))
end

function alloc_fp_state(u::VectorialField, p::VectorialBPM)
    dims = (size(u.Ex)..., size(p.eps_xyz, 3))
    (; amp1 = similar(u.Ex, dims), amp2 = similar(u.Ex, dims))
end

# ---------------------------------------------------------------------------------------
# Pointwise operations, launched through launch_combine! and launch_modal!
# Buffer usage: (bx, by) carry the kick, bx alone carries the longitudinal δ.
# ---------------------------------------------------------------------------------------

inside_cut(kx, ky, kmax2) = kx^2 + ky^2 < kmax2
masked(v, inside::Bool) = ifelse(inside, v, zero(v))
displacement_z(kx, ky, Hx, Hy, k0) = -(kx * Hy - ky * Hx) / k0

# Fourier: masked copy of E⊥ into the kick buffers
struct KickInput{T}
    kmax2::T
end

function (op::KickInput)(kx, ky, ::Direction, ψ, ψ_in)
    Ex, Ey = ψ_in
    inside = inside_cut(kx, ky, op.kmax2)
    (masked(Ex, inside), masked(Ey, inside))
end

# Fourier: pending longitudinal update of E⊥ (bx holds FFT[Δε·Ez]), then kick input
struct KickInputFlush{C, T}
    cL::C
    kmax2::T
end

function (op::KickInputFlush)(kx, ky, ::Direction, ψ, ψ_in)
    Ex, Ey, δ, _ = ψ
    inside = inside_cut(kx, ky, op.kmax2)
    Ex += masked(op.cL * kx * δ, inside)
    Ey += masked(op.cL * ky * δ, inside)
    (Ex, Ey, masked(Ex, inside), masked(Ey, inside))
end

# Real space: ΔHy = c·Δε·Ex and ΔHx = -c·Δε·Ey
struct KickProduct{C, R}
    c::C
    r::R
end

function (op::KickProduct)(ε, ::Direction, ψ, ψ_in)
    bx, by = ψ
    Δε = ε - op.r
    (op.c * Δε * bx, -op.c * Δε * by)
end

# Fourier: H⊥ += ΔH⊥, then D̂z into bx if a longitudinal sheet follows
struct KickOutput{T}
    kmax2::T
    k0::T
    post::Bool
end

function (op::KickOutput)(kx, ky, ::Direction, ψ, ψ_in)
    Hx, Hy, ΔHy = ψ
    ΔHx, = ψ_in
    inside = inside_cut(kx, ky, op.kmax2)
    Hx += masked(ΔHx, inside)
    Hy += masked(ΔHy, inside)
    δ = op.post ? masked(displacement_z(kx, ky, Hx, Hy, op.k0), inside) : ΔHy
    (Hx, Hy, δ)
end

# Real space: δ = Δε·Ez with Ez = Dz/ε
struct LongitudinalProduct{R, T}
    r::R
    nrm::T
end

function (op::LongitudinalProduct)(ε, ::Direction, ψ, ψ_in)
    δ, = ψ
    (op.nrm * δ * (ε - op.r) / ε,)
end

# Fourier: pending longitudinal update, free propagation, then D̂z if a longitudinal
# sheet follows
struct SliceModes{C, T}
    cL::C
    kmax2::T
    k0::T
    pre::Bool
    post::Bool
end

function (op::SliceModes)(m, fac, kx, ky, direction::Direction, ψ, stored)
    Ex, Ey, Hx, Hy, δ = ψ
    inside = inside_cut(kx, ky, op.kmax2)
    if op.pre
        Ex += masked(op.cL * kx * δ, inside)
        Ey += masked(op.cL * ky * δ, inside)
    end
    (Ex, Ey, Hx, Hy), a_out = propagate_modes(m, fac, direction, (Ex, Ey, Hx, Hy),
                                              stored)
    if op.post
        δ = masked(displacement_z(kx, ky, Hx, Hy, op.k0), inside)
    end
    (Ex, Ey, Hx, Hy, δ), a_out
end

# ---------------------------------------------------------------------------------------
# Real-space halves of the two sheets (FFT round trip around a pointwise product)
# ---------------------------------------------------------------------------------------

function kick_product!(p::VectorialBPM, ε, c, direction::Direction)
    bx, by = p.buffers
    p.p_f.ift * bx
    p.p_f.ift * by
    launch_combine!(KickProduct(c, p.eps_ref), (bx, by), (), (ε,), direction)
    p.p_f.ft * bx
    p.p_f.ft * by
end

function longitudinal_product!(p::VectorialBPM, ε, direction::Direction)
    δ = first(p.buffers)
    p.p_f.ift * δ
    launch_combine!(LongitudinalProduct(p.eps_ref, p.nrm_f), (δ,), (), (ε,), direction)
    p.p_f.ft * δ
end

# ---------------------------------------------------------------------------------------
# Lie splitting: K L P forward, P L K backward (exact inverse sequence).
# 5 kernel launches and 6 FFTs per sub-step.
# ---------------------------------------------------------------------------------------

# stored: the slice state at the first sub-step of a slice, () at the other sub-steps
# (full transfer in both directions, no state)
function propagate_substep!(u::VectorialField, stored::Tuple, p::VectorialBPM, ε, cK, cL,
                            k0, direction::Direction)
    bx, by = p.buffers
    fields = (u.Ex, u.Ey, u.Hx, u.Hy, bx)
    k_grid = (p.kx, p.ky)
    modal_coefs = (p.medium.modes, p.factors, p.kx, p.ky)
    if isforward(direction)
        launch_combine!(KickInput(p.kmax2), (bx, by), (u.Ex, u.Ey), k_grid, direction)
        kick_product!(p, ε, cK, direction)
        launch_combine!(KickOutput(p.kmax2, k0, true), (u.Hx, u.Hy, bx), (by,), k_grid,
                        direction)
        longitudinal_product!(p, ε, direction)
        launch_modal!(SliceModes(cL, p.kmax2, k0, true, false), fields, stored,
                      modal_coefs, direction)
    else
        launch_modal!(SliceModes(cL, p.kmax2, k0, false, true), fields, stored,
                      modal_coefs, direction)
        longitudinal_product!(p, ε, direction)
        launch_combine!(KickInputFlush(cL, p.kmax2), (u.Ex, u.Ey, bx, by), (), k_grid,
                        direction)
        kick_product!(p, ε, cK, direction)
        launch_combine!(KickOutput(p.kmax2, k0, false), (u.Hx, u.Hy, bx), (by,), k_grid,
                        direction)
    end
    u
end

function propagate!(u::VectorialField, state, ::Nothing, p::VectorialBPM,
                    direction::Direction)
    T = real(eltype(u.Ex))
    k0 = T(2π / u.lambda)
    cK = Complex{T}(im * sign(direction) * p.dz * k0 * p.nrm_f)  # kick
    cL = Complex{T}(-im * sign(direction) * p.dz / p.eps_ref)    # longitudinal
    for k in reverse(1:size(p.eps_xyz, 3), direction)
        ε = slice_at(p.eps_xyz, k)
        stored = map(a -> slice_at(a, k), state_arrays(state))
        for j in reverse(1:p.n_sub, direction)
            if j == 1
                propagate_substep!(u, stored, p, ε, cK, cL, k0, direction)
            else
                propagate_substep!(u, (), p, ε, cK, cL, k0, direction)
            end
        end
    end
    u
end
