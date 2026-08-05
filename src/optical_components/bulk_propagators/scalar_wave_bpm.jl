struct ScalarWaveBPM{M, K, T, N, P}  <: AbstractBidirectionalComponent{M}
    trainability::Val{M}
    n_xyz::N
    n0::Complex{T}
    n0_loc::Complex{T}
    dz::T
    n_sub::Int
    p_f::P
    kernel::BidirectionalKernel{K}
    kernel_loc::BidirectionalKernel{K}
    conjugate::Bool
    nrm_f::T
end

Functors.@functor ScalarWaveBPM (n_xyz,)

function ScalarWaveBPM(u::ScalarWaveField{U},
                       thickness::Real,
                       n_xyz::AbstractArray{<:Number, 3}, n0::Number;
                       n_sub::Integer = 1,
                       n0_loc::Number = real(n0),
                       trainable::Bool = false,
                       conjugate::Bool = false
                       ) where {T <: Real, U <: AbstractArray{Complex{T}}}
    ns = size(u)[1:2]
    n_slices = size(n_xyz, 3)
    @assert size(n_xyz)[1:2] == ns
    @assert n_slices >= 1
    @assert n_sub >= 1
    dz = T(thickness / (n_slices * n_sub))
    n0 = Complex{T}(n0)
    n0_loc = Complex{T}(n0_loc)
    N = isreal(n_xyz) ? T : Complex{T}
    n_xyz_buf = similar(u.electric, N, size(n_xyz))
    copyto!(n_xyz_buf, n_xyz)
    u_plan = similar(u.electric)
    p_f, _ = make_fft_plans(u_plan, (1, 2); normalize = false)
    kernel = BidirectionalKernel(u, dz, n0; conjugate)
    kernel_loc = BidirectionalKernel(u, dz, n0_loc; conjugate)
    M = trainable ? Trainable : Static
    ScalarWaveBPM(Val(M), n_xyz_buf, n0, n0_loc, dz, n_sub, p_f,
                  kernel, kernel_loc, conjugate, T(1/prod(ns)))
end

trainable(p::ScalarWaveBPM{Trainable}) = (; n_xyz = p.n_xyz)

get_n0(p::ScalarWaveBPM) = p.n0

function alloc_fp_state(u::ScalarWaveField, p::ScalarWaveBPM)
    if p.conjugate
        nothing
    else
        n_slices = size(p.n_xyz, 3)
        (; E_state = similar(u.electric, (size(u.electric)..., n_slices)))
    end
end

function alloc_activations(u, p::ScalarWaveBPM)
    n_slices = size(p.n_xyz, 3)
    (; u = similar(u.electric, (size(u.electric)..., n_slices)))
end

function normalize_fourier(u::ScalarWaveField, p::ScalarWaveBPM)
    @. u.electric *= p.nrm_f
    @. u.electric_dz *= p.nrm_f
end

function propagate_slice!(u::ScalarWaveField, state, activations,
                          p::ScalarWaveBPM, k::Integer; loc::Bool = false)
    backend = get_backend(u.electric)
    n_xy = view(p.n_xyz, :, :, k)
    n0 = loc ? p.n0_loc : p.n0
    kernel = loc ? p.kernel_loc : p.kernel
    E_state = isnothing(state) ? nothing : selectdim(state.E_state, ndims(state.E_state), k)
    u_fwd = isnothing(activations) ? nothing :
        selectdim(activations.u, ndims(activations.u), k)
    normalize_fourier(u, p)
    compute_ift!(p.p_f, u)
    if !isnothing(activations)
        u_fwd = selectdim(activations.u, ndims(activations.u), k)
        copyto!(u_fwd, u.electric)
    end
    @. u.electric_dz += ((2π/u.lambdas.val)^2 * (n0^2 - n_xy^2) * p.dz * u.electric)
    compute_ft!(p.p_f, u)
    propagate_scalar_wave_kernel!(backend)(
        u.electric, u.electric_dz, E_state, kernel, Val(true);
        ndrange = size(u.electric)[1:2])
    u
end

function inverse_propagate_slice!(u::ScalarWaveField, state, activations,
                                  p::ScalarWaveBPM, k::Integer; loc::Bool = false)
    backend = get_backend(u.electric)
    n_xy = view(p.n_xyz, :, :, k)
    n0 = loc ? p.n0_loc : p.n0
    kernel = loc ? p.kernel_loc : p.kernel
    E_state = isnothing(state) ? nothing : selectdim(state.E_state, ndims(state.E_state), k)
    propagate_scalar_wave_kernel!(backend)(
        u.electric, u.electric_dz, E_state, kernel, Val(false);
        ndrange = size(u.electric)[1:2])
    normalize_fourier(u, p)
    compute_ift!(p.p_f, u)
    if !isnothing(activations)
        u_bwd = selectdim(activations.u, ndims(activations.u), k)
        copyto!(u_bwd, u.electric)
    end
    @. u.electric_dz -= ((2π/u.lambdas.val)^2 * (n0^2 - n_xy^2) * p.dz * u.electric)
    compute_ft!(p.p_f, u)
    u
end

function compute_gradient_forward!(∂n_xy, n_xy, u_fwd, ∂u, p::ScalarWaveBPM)
    @. ∂n_xy = -2 * (2π/∂u.lambdas.val)^2 * p.dz * real(conj(n_xy * u_fwd) * ∂u.electric_dz)
end

function propagate_slice_adjoint!(∂u::ScalarWaveField, ∂p,
                                  state, activations, p::ScalarWaveBPM, k::Integer;
                                  loc::Bool = false)
    backend = get_backend(∂u.electric)
    n_xy = view(p.n_xyz, :, :, k)
    n0 = loc ? p.n0_loc : p.n0
    kernel = loc ? p.kernel_loc : p.kernel
    E_state = isnothing(state) ? nothing : selectdim(state.E_state, ndims(state.E_state), k)
    propagate_scalar_wave_adjoint_kernel!(backend)(
        ∂u.electric, ∂u.electric_dz, E_state, kernel, Val(true);
        ndrange = size(∂u.electric)[1:2])
    compute_ift!(p.p_f, ∂u)
    if !isnothing(activations)
        u_fwd = selectdim(activations.u, ndims(activations.u), k)
        ∂n_xy = selectdim(∂p.n_xyz, ndims(∂p.n_xyz), k)
        compute_gradient_forward!(∂n_xy, n_xy, u_fwd, ∂u, p)
    end
    @. ∂u.electric += ((2π/∂u.lambdas.val)^2 * conj(n0^2 - n_xy^2) * p.dz * ∂u.electric_dz)
    compute_ft!(p.p_f, ∂u)
    normalize_fourier(∂u, p)
    ∂u
end

function compute_gradient_backward!(∂n_xy, n_xy, u_bwd, ∂u, p::ScalarWaveBPM)
    @. ∂n_xy = 2 * (2π/∂u.lambdas.val)^2 * p.dz * real(conj(n_xy * u_bwd) * ∂u.electric_dz)
end

function inverse_propagate_slice_adjoint!(∂u::ScalarWaveField, ∂p,
                                          state, activations, p::ScalarWaveBPM, k::Integer;
                                          loc::Bool = false)
    backend = get_backend(∂u.electric)
    n_xy = view(p.n_xyz, :, :, k)
    n0 = loc ? p.n0_loc : p.n0
    kernel = loc ? p.kernel_loc : p.kernel
    E_state = isnothing(state) ? nothing : selectdim(state.E_state, ndims(state.E_state), k)
    compute_ift!(p.p_f, ∂u)
    if !isnothing(activations)
        u_bwd = selectdim(activations.u, ndims(activations.u), k)
        ∂n_xy = selectdim(∂p.n_xyz, ndims(∂p.n_xyz), k)
        compute_gradient_backward!(∂n_xy, n_xy, u_bwd, ∂u, p)
    end
    @. ∂u.electric -= ((2π/∂u.lambdas.val)^2 * conj(n0^2 - n_xy^2) * p.dz * ∂u.electric_dz)
    compute_ft!(p.p_f, ∂u)
    normalize_fourier(∂u, p)
    propagate_scalar_wave_adjoint_kernel!(backend)(
        ∂u.electric, ∂u.electric_dz, E_state, kernel, Val(false);
        ndrange = size(∂u.electric)[1:2])
    ∂u
end

function propagate!(u::ScalarWaveField, state, activations, p::ScalarWaveBPM)
    n_slices = size(p.n_xyz, 3)
    for k in 1:n_slices
        propagate_slice!(u, state, activations, p, k)
        for _ in 1:(p.n_sub - 1)
            propagate_slice!(u, nothing, nothing, p, k; loc = true)
        end
    end
    u
end

function inverse_propagate!(u::ScalarWaveField, state, activations, p::ScalarWaveBPM)
    n_slices = size(p.n_xyz, 3)
    for k in reverse(1:n_slices)
        for _ in 1:(p.n_sub - 1)
            inverse_propagate_slice!(u, nothing, nothing, p, k; loc = true)
        end
        inverse_propagate_slice!(u, state, activations, p, k)
    end
    u
end

function propagate_adjoint!(u::ScalarWaveField, ∂p,
                            state, activations, p::ScalarWaveBPM)
    n_slices = size(p.n_xyz, 3)
    for k in reverse(1:n_slices)
        propagate_slice_adjoint!(u, ∂p, state, activations, p, k)
        for _ in 1:(p.n_sub - 1)
            propagate_slice_adjoint!(u, nothing, nothing, p, k; loc = true)
        end
    end
    u
end

function inverse_propagate_adjoint!(u::ScalarWaveField, ∂p,
                                    state, activations, p::ScalarWaveBPM)
    n_slices = size(p.n_xyz, 3)
    for k in 1:n_slices
        for _ in 1:(p.n_sub - 1)
            inverse_propagate_slice_adjoint!(u, nothing, nothing, p, k; loc = true)
        end
        inverse_propagate_slice_adjoint!(u, ∂p, state, activations, p, k)
    end
    u
end
