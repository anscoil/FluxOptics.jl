struct VectorialBPM{M, T, N, R, P, E, F, K, B} <: AbstractBidirectionalComponent{M}
    trainability::Val{M}
    eps_xyz::N
    eps_ref::R
    dz::T
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
                      trainable::Bool = false)
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
    kx, ky = similar(u.Ex, T, (ns[1], 1)), similar(u.Ex, T, (1, ns[2]))
    copyto!(kx, collect(T, T(2π) .* fftfreq(ns[1], 1 / u.ds[1])))
    copyto!(ky, collect(T, T(2π) .* fftfreq(ns[2], 1 / u.ds[2])))
    buffers = (similar(u.Ex), similar(u.Ex))
    M = trainable ? Trainable : Static
    VectorialBPM(Val(M), eps_buf, ε_r, dz, p_f, medium, factors, kx, ky, buffers,
                 T(1 / prod(ns)))
end

function alloc_fp_state(u::VectorialField, p::VectorialBPM)
    dims = (size(u.Ex)..., size(p.eps_xyz, 3))
    (; amp1 = similar(u.Ex, dims), amp2 = similar(u.Ex, dims))
end

# function propagate_slice_direct!(u::VectorialField, p::VectorialBPM, k::Integer,
#                                  direction::Direction)
#     T = real(eltype(u.Ex))
#     ε, ε_r = view(p.eps_xyz, :, :, k), p.eps_ref
#     c = im * sign(direction) * p.dz * T(2π / u.lambda) * p.nrm_f
#     bx, by = p.buffers
#     copyto!(bx, u.Ex)
#     copyto!(by, u.Ey)
#     p.p_f.ift * bx
#     p.p_f.ift * by
#     @. bx = c * (ε - ε_r) * bx     # ΔHy
#     @. by = -c * (ε - ε_r) * by    # ΔHx
#     p.p_f.ft * bx
#     p.p_f.ft * by
#     u.Hx .+= by
#     u.Hy .+= bx
#     u
# end

# function propagate_slice_longitudinal!(u::VectorialField, p::VectorialBPM, k::Integer,
#                                        direction::Direction)
#     T = real(eltype(u.Ex))
#     k0 = T(2π / u.lambda)
#     ε, ε_r = view(p.eps_xyz, :, :, k), p.eps_ref
#     δz = first(p.buffers)
#     @. δz = -(p.kx * u.Hy - p.ky * u.Hx) / k0    # D̂z
#     p.p_f.ift * δz
#     @. δz *= p.nrm_f * (ε - ε_r) / ε             # δz = Δε·Ez, Ez = Dz/ε
#     p.p_f.ft * δz
#     c = -im * sign(direction) * p.dz / ε_r
#     @. u.Ex += c * p.kx * δz
#     @. u.Ey += c * p.ky * δz
#     u
# end

function propagate_slice_direct!(u::VectorialField, p::VectorialBPM, k::Integer,
                                 direction::Direction; k_cut::Real = 40)
    T = real(eltype(u.Ex))
    k0 = T(2π / u.lambda)
    kc2 = T(k_cut * k0)^2
    ε, ε_r = slice_at(p.eps_xyz, k), p.eps_ref
    c = im * sign(direction) * p.dz * k0 * p.nrm_f
    bx, by = p.buffers
    z = zero(eltype(bx))
    @. bx = ifelse(p.kx^2 + p.ky^2 < kc2, u.Ex, z)
    @. by = ifelse(p.kx^2 + p.ky^2 < kc2, u.Ey, z)
    p.p_f.ift * bx
    p.p_f.ift * by
    @. bx = c * (ε - ε_r) * bx     # ΔHy
    @. by = -c * (ε - ε_r) * by    # ΔHx
    p.p_f.ft * bx
    p.p_f.ft * by
    @. u.Hx += ifelse(p.kx^2 + p.ky^2 < kc2, by, z)
    @. u.Hy += ifelse(p.kx^2 + p.ky^2 < kc2, bx, z)
    u
end

function propagate_slice_longitudinal!(u::VectorialField, p::VectorialBPM, k::Integer,
                                       direction::Direction; k_cut::Real = 40)
    T = real(eltype(u.Ex))
    k0 = T(2π / u.lambda)
    kc2 = T(k_cut * k0)^2
    ε, ε_r = slice_at(p.eps_xyz, k), p.eps_ref
    δz = first(p.buffers)
    z = zero(eltype(δz))
    @. δz = ifelse(p.kx^2 + p.ky^2 < kc2, -(p.kx * u.Hy - p.ky * u.Hx) / k0, z)
    p.p_f.ift * δz
    @. δz *= p.nrm_f * (ε - ε_r) / ε             # δz = Δε·Ez, Ez = Dz/ε
    p.p_f.ft * δz
    c = -im * sign(direction) * p.dz / ε_r
    @. u.Ex += ifelse(p.kx^2 + p.ky^2 < kc2, c * p.kx * δz, z)
    @. u.Ey += ifelse(p.kx^2 + p.ky^2 < kc2, c * p.ky * δz, z)
    u
end

function propagate_slice_fourier!(u::VectorialField, state, p::VectorialBPM, k::Integer,
                                  direction::Direction)
    stored = map(a -> slice_at(a, k), state_arrays(state))
    launch_modal!(propagate_modes, (u.Ex, u.Ey, u.Hx, u.Hy), stored,
                  (p.medium.modes, p.factors), direction)
end

function propagate_slice!(u::VectorialField, state, p::VectorialBPM, k::Integer,
                          ::Forward)
    propagate_slice_direct!(u, p, k, Forward())
    propagate_slice_longitudinal!(u, p, k, Forward())
    propagate_slice_fourier!(u, state, p, k, Forward())
    u
end

function propagate_slice!(u::VectorialField, state, p::VectorialBPM, k::Integer,
                          ::Backward)
    propagate_slice_fourier!(u, state, p, k, Backward())
    propagate_slice_longitudinal!(u, p, k, Backward())
    propagate_slice_direct!(u, p, k, Backward())
    u
end

function propagate!(u::VectorialField, state, ::Nothing, p::VectorialBPM,
                    direction::Direction)
    for k in reverse(1:size(p.eps_xyz, 3), direction)
        propagate_slice!(u, state, p, k, direction)
    end
    u
end
