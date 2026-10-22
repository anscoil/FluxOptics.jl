struct VectorialBPM{M, T, N, R, P, E, F, K, B} <: AbstractBidirectionalComponent{M}
    trainability::Val{M}
    eps_xyz::N
    eps_ref::R
    dz::T
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
                      k_max::Real = Inf, trainable::Bool = false)
    T = real(eltype(u.Ex))
    ns = size(u.Ex)[1:2]
    @assert size(eps_xyz)[1:2] == ns
    dz = T(thickness / size(eps_xyz, 3))
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
    VectorialBPM(Val(M), eps_buf, ε_r, dz, T(k_max)^2, p_f, medium, factors, kx, ky,
                 buffers, T(1 / prod(ns)))
end

function alloc_fp_state(u::VectorialField, p::VectorialBPM)
    dims = (size(u.Ex)..., size(p.eps_xyz, 3))
    (; amp1 = similar(u.Ex, dims), amp2 = similar(u.Ex, dims))
end

# ---------------------------------------------------------------------------------------
# One slice applies three operators, each over the full step dz (Lie splitting):
#   K (kick):         H⊥ += cK·Π[Δε·Π[E⊥]]
#   L (longitudinal): E⊥ += k⊥·Π[cL·(Δε/ε)·Π[D̂z]],  D̂z = -(kx·Hy - ky·Hx)/k0
#   P (propagation):  modal propagation in the reference medium ε_r
# with Δε = ε - ε_r and Π[·] keeping the transverse frequencies |k⊥| < k_max.
# K and L are shears (one pair of components updated from the other), so their inverse
# is the same operator with the opposite coefficient: sign(direction) in cK and cL.
# Forward: K, L, P. Backward: P, L, K, the exact inverse sequence.
#
# Both sheets follow the same pattern:
#   Input (Fourier) → IFFT → Product (real space) → FFT → Output (Fourier)
# The kick uses both buffers (bx, by), the longitudinal sheet uses bx only.
# ---------------------------------------------------------------------------------------

inside_cut(kx, ky, kmax2) = kx^2 + ky^2 < kmax2
masked(v, inside::Bool) = ifelse(inside, v, zero(v))

# Kick input, Fourier: (bx, by) = Π[E⊥]
struct KickInput{T}
    kmax2::T
end

function (op::KickInput)(kx, ky, ::Direction, ψ, ψ_in)
    Ex, Ey = ψ_in
    inside = inside_cut(kx, ky, op.kmax2)
    (masked(Ex, inside), masked(Ey, inside))
end

# Kick product, real space: (bx, by) holds E⊥ on input and ΔH⊥ on output,
# ΔHx = -c·Δε·Ey and ΔHy = c·Δε·Ex
struct KickProduct{C, R}
    c::C
    r::R
end

function (op::KickProduct)(ε, ::Direction, ψ, ψ_in)
    Ex, Ey = ψ
    Δε = ε - op.r
    (-op.c * Δε * Ey, op.c * Δε * Ex)
end

# Kick output, Fourier: H⊥ += Π[ΔH⊥]
struct KickOutput{T}
    kmax2::T
end

function (op::KickOutput)(kx, ky, ::Direction, ψ, ψ_in)
    Hx, Hy = ψ
    ΔHx, ΔHy = ψ_in
    inside = inside_cut(kx, ky, op.kmax2)
    (Hx + masked(ΔHx, inside), Hy + masked(ΔHy, inside))
end

# Longitudinal input, Fourier: bx = Π[D̂z]
struct LongitudinalInput{T}
    kmax2::T
    k0::T
end

function (op::LongitudinalInput)(kx, ky, ::Direction, ψ, ψ_in)
    Hx, Hy = ψ_in
    inside = inside_cut(kx, ky, op.kmax2)
    (masked(-(kx * Hy - ky * Hx) / op.k0, inside),)
end

# Longitudinal product, real space: bx = c·(Δε/ε)·D̂z, i.e. c·Δε·Ez
struct LongitudinalProduct{C, R}
    c::C
    r::R
end

function (op::LongitudinalProduct)(ε, ::Direction, ψ, ψ_in)
    δ, = ψ
    (op.c * δ * (ε - op.r) / ε,)
end

# Longitudinal output, Fourier: E⊥ += k⊥·Π[bx]
struct LongitudinalOutput{T}
    kmax2::T
end

function (op::LongitudinalOutput)(kx, ky, ::Direction, ψ, ψ_in)
    Ex, Ey = ψ
    δ, = ψ_in
    δ = masked(δ, inside_cut(kx, ky, op.kmax2))
    (Ex + kx * δ, Ey + ky * δ)
end

# ---------------------------------------------------------------------------------------
# The three operators
# ---------------------------------------------------------------------------------------

# K: 3 launches, 4 FFTs
function kick!(u::VectorialField, p::VectorialBPM, ε, cK, direction::Direction)
    bx, by = p.buffers
    k_grid = (p.kx, p.ky)
    launch_combine!(KickInput(p.kmax2), (bx, by), (u.Ex, u.Ey), k_grid, direction)
    p.p_f.ift * bx
    p.p_f.ift * by
    launch_combine!(KickProduct(cK, p.eps_ref), (bx, by), (), (ε,), direction)
    p.p_f.ft * bx
    p.p_f.ft * by
    launch_combine!(KickOutput(p.kmax2), (u.Hx, u.Hy), (bx, by), k_grid, direction)
    u
end

# L: 3 launches, 2 FFTs
function longitudinal!(u::VectorialField, p::VectorialBPM, ε, cL, k0,
                       direction::Direction)
    δ = first(p.buffers)
    k_grid = (p.kx, p.ky)
    launch_combine!(LongitudinalInput(p.kmax2, k0), (δ,), (u.Hx, u.Hy), k_grid, direction)
    p.p_f.ift * δ
    launch_combine!(LongitudinalProduct(cL, p.eps_ref), (δ,), (), (ε,), direction)
    p.p_f.ft * δ
    launch_combine!(LongitudinalOutput(p.kmax2), (u.Ex, u.Ey), (δ,), k_grid, direction)
    u
end

# P: 1 launch, stored is the state of the slice
function free_propagation!(u::VectorialField, stored::Tuple, p::VectorialBPM,
                           direction::Direction)
    launch_modal!(propagate_modes, (u.Ex, u.Ey, u.Hx, u.Hy), stored,
                  (p.medium.modes, p.factors), direction)
    u
end

# ---------------------------------------------------------------------------------------
# Slice and slice loop
# ---------------------------------------------------------------------------------------

@inline function propagate_slice!(u::VectorialField, stored::Tuple, p::VectorialBPM, ε,
                                  cK, cL, k0, direction::Direction)
    if isforward(direction)
        kick!(u, p, ε, cK, direction)
        longitudinal!(u, p, ε, cL, k0, direction)
        free_propagation!(u, stored, p, direction)
    else
        free_propagation!(u, stored, p, direction)
        longitudinal!(u, p, ε, cL, k0, direction)
        kick!(u, p, ε, cK, direction)
    end
    u
end

function propagate!(u::VectorialField, state, ::Nothing, p::VectorialBPM,
                    direction::Direction)
    T = real(eltype(u.Ex))
    k0 = T(2π / u.lambda)
    # Sheet coefficients: nrm_f normalizes the FFT round trip inside each sheet, and
    # sign(direction) turns each sheet into its inverse
    cK = Complex{T}(im * sign(direction) * p.dz * k0 * p.nrm_f)
    cL = Complex{T}(-im * sign(direction) * p.dz * p.nrm_f / p.eps_ref)
    for k in reverse(1:size(p.eps_xyz, 3), direction)
        ε = slice_at(p.eps_xyz, k)
        stored = map(a -> slice_at(a, k), state_arrays(state))
        propagate_slice!(u, stored, p, ε, cK, cL, k0, direction)
    end
    u
end
