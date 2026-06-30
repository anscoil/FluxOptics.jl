struct ScalarWaveBPM{M, K, T, N, P}  <: AbstractBidirectionalComponent{M}
    trainability::Val{M}
    n_xyz::N
    n0::Complex{T}
    dz::T
    p_f::P
    kernel::BidirectionalKernel{K}
    conjugate::Bool
end

Functors.@functor ScalarWaveBPM (n_xyz,)

function ScalarWaveBPM(u::ScalarWaveField{U},
                       thickness::Real,
                       n_xyz::AbstractArray{<:Number, 3}, n0::Number;
                       trainable::Bool = false,
                       conjugate::Bool = false) where {T <: Real, U <: AbstractArray{Complex{T}}}
    ns = size(u)[1:2]
    n_slices = size(n_xyz, 3)
    @assert size(n_xyz)[1:2] == ns
    @assert n_slices >= 2
    dz = T(thickness / n_slices)
    n0 = Complex{T}(n0)
    N = isreal(n_xyz) ? T : Complex{T}
    n_xyz_buf = similar(u.electric, N, size(n_xyz))
    copyto!(n_xyz_buf, n_xyz)
    u_plan = similar(u.electric)
    p_f, _ = make_fft_plans(u_plan, (1, 2); normalize = true)
    kernel = BidirectionalKernel(u, dz, n0; conjugate)
    M = trainable ? Trainable : Static
    ScalarWaveBPM(Val(M), n_xyz_buf, n0, dz, p_f, kernel, conjugate)
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
    (; u_fwd = similar(u.electric, (size(u.electric)..., n_slices)),
     u_bwd = similar(u.electric, (size(u.electric)..., n_slices)))
end

function propagate_slice!(u::ScalarWaveField, state, activations,
                          p::ScalarWaveBPM, k::Integer)
    backend = get_backend(u.electric)
    n_xy = view(p.n_xyz, :, :, k)
    E_state = isnothing(state) ? nothing : selectdim(state.E_state, ndims(state.E_state), k)
    u_fwd = isnothing(activations) ? nothing :
        selectdim(activations.u_fwd, ndims(activations.u_fwd), k)
    compute_ift!(p.p_f, u)
    if !isnothing(activations)
        u_fwd = selectdim(activations.u_fwd, ndims(activations.u_fwd), k)
        copyto!(u_fwd, u.electric)
    end
    @. u.electric_dz += ((2π/u.lambdas.val)^2 * (p.n0^2 - n_xy^2) * p.dz * u.electric)
    compute_ft!(p.p_f, u)
    propagate_scalar_wave_kernel!(backend)(
        u.electric, u.electric_dz, E_state, p.kernel, Val(true);
        ndrange = size(u.electric)[1:2])
    u
end

function inverse_propagate_slice!(u::ScalarWaveField, state, activations,
                                  p::ScalarWaveBPM, k::Integer)
    backend = get_backend(u.electric)
    n_xy = view(p.n_xyz, :, :, k)
    E_state = isnothing(state) ? nothing : selectdim(state.E_state, ndims(state.E_state), k)
    propagate_scalar_wave_kernel!(backend)(
        u.electric, u.electric_dz, E_state, p.kernel, Val(false);
        ndrange = size(u.electric)[1:2])
    compute_ift!(p.p_f, u)
    if !isnothing(activations)
        u_bwd = selectdim(activations.u_bwd, ndims(activations.u_bwd), k)
        copyto!(u_bwd, u.electric)
    end
    @. u.electric_dz -= ((2π/u.lambdas.val)^2 * (p.n0^2 - n_xy^2) * p.dz * u.electric)
    compute_ft!(p.p_f, u)
    u
end

function propagate_slice_adjoint!(u::ScalarWaveField, state, p::ScalarWaveBPM, k::Integer)
    backend = get_backend(u.electric)
    n_xy = view(p.n_xyz, :, :, k)
    E_state = isnothing(state) ? nothing : selectdim(state.E_state, ndims(state.E_state), k)
    propagate_scalar_wave_adjoint_kernel!(backend)(
        u.electric, u.electric_dz, E_state, p.kernel, Val(true);
        ndrange = size(u.electric)[1:2])
    compute_ift!(p.p_f, u)
    @. u.electric += ((2π/u.lambdas.val)^2 * conj(p.n0^2 - n_xy^2) * p.dz * u.electric_dz)
    compute_ft!(p.p_f, u)
    u
end

function inverse_propagate_slice_adjoint!(u::ScalarWaveField, state,
                                          p::ScalarWaveBPM, k::Integer)
    backend = get_backend(u.electric)
    n_xy = view(p.n_xyz, :, :, k)
    E_state = isnothing(state) ? nothing : selectdim(state.E_state, ndims(state.E_state), k)
    compute_ift!(p.p_f, u)
    @. u.electric -= ((2π/u.lambdas.val)^2 * conj(p.n0^2 - n_xy^2) * p.dz * u.electric_dz)
    compute_ft!(p.p_f, u)
    propagate_scalar_wave_adjoint_kernel!(backend)(
        u.electric, u.electric_dz, E_state, p.kernel, Val(false);
        ndrange = size(u.electric)[1:2])
    u
end

function propagate!(u::ScalarWaveField, state, activations, p::ScalarWaveBPM)
    n_slices = size(p.n_xyz, 3)
    for k in 1:n_slices
        propagate_slice!(u, state, activations, p, k)
    end
    u
end

function inverse_propagate!(u::ScalarWaveField, state, activations, p::ScalarWaveBPM)
    n_slices = size(p.n_xyz, 3)
    for k in reverse(1:n_slices)
        inverse_propagate_slice!(u, state, activations, p, k)
    end
    u
end

function propagate_adjoint!(u::ScalarWaveField, ::Nothing,
                            state, ::Nothing,
                            p::ScalarWaveBPM)
    n_slices = size(p.n_xyz, 3)
    for k in reverse(1:n_slices)
        propagate_slice_adjoint!(u, state, p, k)
    end
    u
end

function inverse_propagate_adjoint!(u::ScalarWaveField, ::Nothing,
                                    state, ::Nothing, p::ScalarWaveBPM)
    n_slices = size(p.n_xyz, 3)
    for k in 1:n_slices
        inverse_propagate_slice_adjoint!(u, state, p, k)
    end
    u
end
