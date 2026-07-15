struct ScalarWaveBiProp{M, K, A, E, P, U, T}  <: AbstractBidirectionalComponent{M}
    trainability::Val{M}
    mask_xyz::A
    mask_eps::E
    n1::Complex{T}
    n2::Complex{T}
    dz::T
    kernel_n1::BidirectionalKernel{K}
    kernel_n2::BidirectionalKernel{K}
    u_tmp::ScalarWaveField{U}
    p_f::P
end

function ScalarWaveBiProp(u::ScalarWaveField{U}, thickness::Real,
                          mask_xyz::AbstractArray{<:Number, 3}, n1::Number, n2::Number;
                          mask_eps = nothing
                          ) where {T <: Real, U <: AbstractArray{Complex{T}}}
    ns = size(u)[1:2]
    n_slices = size(mask_xyz, 3)
    @assert size(mask_xyz)[1:2] == ns
    @assert n_slices >= 1
    dz = T(thickness / n_slices)
    kernel_n1 = BidirectionalKernel(u, dz, n1; conjugate = true)
    kernel_n2 = BidirectionalKernel(u, dz, n2; conjugate = true)
    n1 = Complex{T}(n1)
    n2 = Complex{T}(n2)
    mask_buf = similar(u.electric, T, size(mask_xyz))
    copyto!(mask_buf, mask_xyz)
    mask_eps_buf = similar(u.electric, Complex{T}, size(mask_xyz))
    if !isnothing(mask_eps)
        copyto!(mask_eps_buf, mask_eps)
    else
        @. mask_eps_buf = n1^2 * mask_buf + n2^2 * (1 - mask_buf)
    end
    u_tmp = similar(u)
    u_plan = similar(u.electric)
    p_f, _ = make_fft_plans(u_plan, (1, 2); normalize = true)
    ScalarWaveBiProp(Val(Static), mask_buf, mask_eps_buf, n1, n2, dz,
                     kernel_n1, kernel_n2, u_tmp, p_f)
end

get_n0(p::ScalarWaveBiProp) = nothing

alloc_fp_state(u::ScalarWaveField, p::ScalarWaveBiProp) = nothing

function apply_mask!(u::AbstractArray, mask, conjugate::Bool = false)
    if !conjugate
        @. u *= mask
    else
        @. u *= 1 - mask
    end
end

function apply_mask!(u::ScalarWaveField, p::ScalarWaveBiProp,
                     k::Integer, conjugate::Bool = false)
    mask = view(p.mask_xyz, :, :, k)
    apply_mask!(u.electric, mask, conjugate)
    apply_mask!(u.electric_dz, mask, conjugate)
end

function apply_correction!(u::ScalarWaveField, p::ScalarWaveBiProp,
                           k::Integer, conjugate::Bool = false)
    mask_eps = view(p.mask_eps, :, :, k)
    if !conjugate
        @. u.electric_dz += ((2π/u.lambdas.val)^2 * (p.n1^2 - mask_eps) * p.dz * u.electric)
        # @. u.electric *= cis(2π/u.lambdas.val * (1 - mask_bin) * (p.n2 - p.n1) * p.dz)
        # @. u.electric_dz *= cis(2π/u.lambdas.val * (1 - mask_bin) * (p.n2 - p.n1) * p.dz)
    else
        @. u.electric_dz += ((2π/u.lambdas.val)^2 * (p.n2^2 - mask_eps) *  p.dz * u.electric)
        # @. u.electric *= cis(2π/u.lambdas.val * mask_bin * (p.n1 - p.n2) * p.dz)
        # @. u.electric_dz *= cis(2π/u.lambdas.val * mask_bin * (p.n1 - p.n2) * p.dz)
    end
end

function inverse_apply_correction!(u::ScalarWaveField, p::ScalarWaveBiProp,
                                   k::Integer, conjugate::Bool = false)
    mask_eps = view(p.mask_eps, :, :, k)
    if !conjugate
        @. u.electric_dz -= ((2π/u.lambdas.val)^2 * (p.n1^2 - mask_eps) * p.dz * u.electric)
        # @. u.electric *= cis(-2π/u.lambdas.val * (1 - mask_bin) * (p.n2 - p.n1) * p.dz)
        # @. u.electric_dz *= cis(-2π/u.lambdas.val * (1 - mask_bin) * (p.n2 - p.n1) * p.dz)
    else
        @. u.electric_dz -= ((2π/u.lambdas.val)^2 * (p.n2^2 - mask_eps) * p.dz * u.electric)
        # @. u.electric *= cis(-2π/u.lambdas.val * mask_bin * (p.n1 - p.n2) * p.dz)
        # @. u.electric_dz *= cis(-2π/u.lambdas.val * mask_bin * (p.n1 - p.n2) * p.dz)
    end
end

function add!(u::ScalarWaveField, v::ScalarWaveField)
    @. u.electric += v.electric
    @. u.electric_dz += v.electric_dz
end

function sub!(u::ScalarWaveField, v::ScalarWaveField)
    @. u.electric -= v.electric
    @. u.electric_dz -= v.electric_dz
end

function keep_forward!(u::ScalarWaveField, a)
    @. u.electric = 0.5 * (u.electric + u.electric_dz / a)
    @. u.electric_dz = a * u.electric
    u
end

function keep_backward!(u::ScalarWaveField, a)
    @. u.electric = 0.5 * (u.electric - u.electric_dz / a)
    @. u.electric_dz = -a * u.electric
    u
end

@kernel function propagate_binary_kernel!(e1, e1_dz,
                                          e2, e2_dz,
                                          kernel_n1, kernel_n2,
                                          ::Val{forward}) where {forward}
    s = forward ? 1 : -1
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(e1)[3:end])
        a1 = _get_val(kernel_n1.a, I, J)
        a2 = _get_val(kernel_n2.a, I, J)
        exp_a1_p = _get_val(kernel_n1.exp_a_p, I, J)
        exp_a1_m = _get_val(kernel_n1.exp_a_m, I, J)
        exp_a2_p = _get_val(kernel_n2.exp_a_p, I, J)
        exp_a2_m = _get_val(kernel_n2.exp_a_m, I, J)
        E1_val = e1[I,J]
        dE1_val = e1_dz[I,J]
        E2_val = e2[I,J]
        dE2_val = e2_dz[I,J]
        E1 = 0.5 * (E1_val + s * dE1_val / a1) * exp_a1_p
        E2 = 0.5 * (E1_val - s * dE1_val / a1) * exp_a1_m
        E3 = 0.5 * (E2_val + s * dE2_val / a2) * exp_a2_p
        E4 = 0.5 * (E2_val - s * dE2_val / a2) * exp_a2_m
        e1[I,J] = E1 + E2 + E3 + E4
        e1_dz[I,J] = s * (a1 * (E1 - E2) + a2 * (E3 - E4))
    end
end

function propagate_slice!(u::ScalarWaveField, state, activations,
                          p::ScalarWaveBiProp, k::Integer)
    backend = get_backend(u.electric)

    compute_ift!(p.p_f, u)
    copyto!(p.u_tmp, u)
    v = p.u_tmp

    apply_mask!(u, p, k)
    apply_mask!(v, p, k, true)
    apply_correction!(u, p, k)
    apply_correction!(v, p, k, true)
    compute_ft!(p.p_f, u)
    compute_ft!(p.p_f, v)
    
    propagate_binary_kernel!(backend)(
        u.electric, u.electric_dz,
        v.electric, v.electric_dz,
        p.kernel_n1, p.kernel_n2, Val(true);
        ndrange = size(u.electric)[1:2])

    # propagate_scalar_wave_kernel!(backend)(
    #     u.electric, u.electric_dz, nothing, p.kernel_n1, Val(true);
    #     ndrange = size(u.electric)[1:2])

    # propagate_scalar_wave_kernel!(backend)(
    #     v.electric, v.electric_dz, nothing, p.kernel_n2, Val(true);
    #     ndrange = size(u.electric)[1:2])

    # add!(u, v)
        
    u
end

function inverse_propagate_slice!(u::ScalarWaveField, state, activations,
                                  p::ScalarWaveBiProp, k::Integer)
    backend = get_backend(u.electric)

    compute_ift!(p.p_f, u)
    copyto!(p.u_tmp, u)
    v = p.u_tmp

    apply_mask!(u, p, k)
    apply_mask!(v, p, k, true)
    inverse_apply_correction!(u, p, k)
    inverse_apply_correction!(v, p, k, true)
    compute_ft!(p.p_f, u)
    compute_ft!(p.p_f, v)
    
    propagate_binary_kernel!(backend)(
        u.electric, u.electric_dz,
        v.electric, v.electric_dz,
        p.kernel_n1, p.kernel_n2, Val(false);
        ndrange = size(u.electric)[1:2])

    # propagate_scalar_wave_kernel!(backend)(
    #     u.electric, u.electric_dz, nothing, p.kernel_n1, Val(false);
    #     ndrange = size(u.electric)[1:2])

    # propagate_scalar_wave_kernel!(backend)(
    #     v.electric, v.electric_dz, nothing, p.kernel_n2, Val(false);
    #     ndrange = size(u.electric)[1:2])

    # add!(u, v)
    
    u
end

function propagate!(u::ScalarWaveField, state, activations, p::ScalarWaveBiProp)
    n_slices = size(p.mask_xyz, 3)
    for k in 1:n_slices
        propagate_slice!(u, state, activations, p, k)
    end
    u
end

function inverse_propagate!(u::ScalarWaveField, state, activations, p::ScalarWaveBiProp)
    n_slices = size(p.mask_xyz, 3)
    for k in reverse(1:n_slices)
        inverse_propagate_slice!(u, state, activations, p, k)
    end
    u
end
