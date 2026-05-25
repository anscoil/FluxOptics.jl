struct BidirectionalKernel{K, T}
    a::K
    r01::K
    t10::K
    r02::K
    exp_a_p::K
    exp_a_m::K
    nrm_f::T
end

Adapt.@adapt_structure BidirectionalKernel

struct BidirectionalBPM{M, N, K, E, U, T, P}  <: AbstractCustomComponent{M}
    trainability::Val{M}
    n_xyz::N
    n0::Complex{T}
    n1::Complex{T}
    n2::Complex{T}
    dz::T
    p_f::P
    E_tmp::E
    ub::HelmholtzField{U}
    β::Ref{T}
    kernel::BidirectionalKernel{K, T}
end

Functors.@functor BidirectionalBPM (n_xyz,)

function optimal_gauge(n_xyz, fill_factor=0.5)
    re_n2 = real.(n_xyz.^2)
    n0_real = sqrt(fill_factor * maximum(re_n2) + (1-fill_factor) * minimum(re_n2))
    n0_imag = 0.05 * n0_real
    return complex(n0_real, n0_imag)
end

function BidirectionalBPM(u::HelmholtzField{U},
                          thickness::Real,
                          n_xyz::AbstractArray{<:Number, 3};
                          n0::Number = optimal_gauge(n_xyz),
                          n1::Number = 1.0,
                          n2::Number = 1.0,
                          trainable::Bool = false,
                          buffered::Bool = false) where {T <: Real,
                                                         U <: AbstractArray{Complex{T}}}
    ns = size(u)[1:2]
    n_slices = size(n_xyz, 3)
    @assert size(n_xyz)[1:2] == ns
    @assert n_slices >= 2
    dz = T(thickness / n_slices)
    M  = trainability(trainable, buffered)
    N = isreal(n_xyz) ? T : Complex{T}
    n_xyz_buf = similar(u.electric, N, size(n_xyz))
    copyto!(n_xyz_buf, n_xyz)
    n0 = Complex{T}(n0)
    n1 = Complex{T}(n1)
    n2 = Complex{T}(n2)
    a = im * compute_kz(u, n0)
    r01 = compute_fresnel_r12(u, n0, n1)
    t10 = compute_fresnel_t12(u, n1, n0)
    r02 = compute_fresnel_r12(u, n0, n2)
    exp_a_p = @. exp(a * dz)
    exp_a_m = @. exp(-a * dz)
    u_plan = similar(u.electric)
    p_f = make_fft_plans(u_plan, (1, 2); normalize = false)
    E_tmp = similar(u.electric, (size(u.electric)..., n_slices))
    @. E_tmp = 0
    ub = similar(u)
    @. ub.electric = 0
    @. ub.electric_dz = 0
    nrm_f = T(1/prod(ns))
    kernel = BidirectionalKernel(a, r01, t10, r02, exp_a_p, exp_a_m, nrm_f)
    BidirectionalBPM(Val(M), n_xyz_buf, n0, n1, n2, dz, p_f, E_tmp, ub, Ref(T(0)), kernel)
end

_get_val(A::AbstractArray{<:Any, 2}, I, J) = A[I]
function _get_val(A::AbstractArray{<:Any, N}, I, J) where {N}
    A[I, CartesianIndex(min.(Tuple(J), size(A)[3:end]))]
end

@kernel function propagate_helmholtz_forward_kernel!(electric, electric_dz, E_tmp, kernel, β)
    nrm_f = kernel.nrm_f
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(electric)[3:end])
        a = _get_val(kernel.a, I, J)
        exp_a_p = _get_val(kernel.exp_a_p, I, J)
        exp_a_m = _get_val(kernel.exp_a_m, I, J)
        E_val = electric[I, J]
        dE_val = electric_dz[I, J]
        E_minus = (1-β) * E_tmp[I, J]
        if β > 0
            E_minus += 0.5 * β * (E_val - dE_val / a)
        end
        E_plus = 0.5 * (E_val + dE_val / a)
        E_plus *= exp_a_p
        E_minus *= exp_a_m
        electric[I,J] = nrm_f * (E_plus + E_minus)
        electric_dz[I,J] = nrm_f * a * (E_plus - E_minus)
        E_tmp[I, J] = E_plus
    end
end

@kernel function propagate_helmholtz_backward_kernel!(electric, electric_dz, E_tmp, kernel, β)
    nrm_f = kernel.nrm_f
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(electric)[3:end])
        a = _get_val(kernel.a, I, J)
        exp_a_p = _get_val(kernel.exp_a_p, I, J)
        exp_a_m = _get_val(kernel.exp_a_m, I, J)
        E_val = electric[I, J]
        dE_val = electric_dz[I, J]
        E_plus = (1-β) * E_tmp[I, J]
        if β > 0
            E_plus += 0.5 * β * (E_val + dE_val / a)
        end
        E_minus = 0.5 * (E_val - dE_val / a)
        E_plus *= exp_a_m
        E_minus *= exp_a_p
        electric[I,J] = nrm_f * (E_plus + E_minus)
        electric_dz[I,J] = nrm_f * a * (E_plus - E_minus)
        E_tmp[I, J] = E_minus
    end
end

@kernel function propagate_helmholtz_kernel!(electric, electric_dz, E_tmp,
                                             kernel, β, ::Val{forward}) where {forward}
    nrm_f = kernel.nrm_f
    s = forward ? 1 : -1
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(electric)[3:end])
        a = _get_val(kernel.a, I, J)
        exp_a_p = _get_val(kernel.exp_a_p, I, J)
        exp_a_m = _get_val(kernel.exp_a_m, I, J)
        E_val = electric[I, J]
        dE_val = electric_dz[I, J]
        E2 = (1-β) * E_tmp[I, J]
        if β > 0
            E2 += 0.5 * β * (E_val - s * dE_val / a)
        end
        E1 = 0.5 * (E_val + s * dE_val / a)
        E1 *= exp_a_p
        E2 *= exp_a_m # conj(exp_a_p)
        electric[I,J] = nrm_f * (E1 + E2)
        electric_dz[I,J] = nrm_f * a * s * (E1 - E2)
        E_tmp[I,J] = E1
    end
end

function propagate_slice_forward!(u::HelmholtzField, p::BidirectionalBPM, k::Integer)
    β = p.β[]
    @assert 0 <= β <= 1
    backend = get_backend(u.electric)
    n_xy = view(p.n_xyz, :, :, k)
    @. u.electric_dz += ((2π/u.lambdas.val)^2 * (p.n0^2 - n_xy^2) * p.dz * u.electric)
    compute_ft!(p.p_f, u)
    E_minus = selectdim(p.E_tmp, ndims(p.E_tmp), k)
    propagate_helmholtz_kernel!(backend)(
        u.electric, u.electric_dz, E_minus, p.kernel, β, Val(true);
        ndrange = size(u.electric)[1:2])
    compute_ift!(p.p_f, u)
end

function propagate_slice_backward!(u::HelmholtzField, p::BidirectionalBPM, k::Integer)
    β = p.β[]
    @assert 0 <= β <= 1
    backend = get_backend(u.electric)
    n_xy = view(p.n_xyz, :, :, k)
    compute_ft!(p.p_f, u)
    E_plus = selectdim(p.E_tmp, ndims(p.E_tmp), k)
    propagate_helmholtz_kernel!(backend)(
        u.electric, u.electric_dz, E_plus, p.kernel, β, Val(false);
        ndrange = size(u.electric)[1:2])
    compute_ift!(p.p_f, u)
    @. u.electric_dz -= ((2π/u.lambdas.val)^2 * (p.n0^2 - n_xy^2) * p.dz * u.electric)
end

@kernel function boundary_condition_kernel_1!(electric, electric_dz, E_f, kernel)
    nrm_f = kernel.nrm_f
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(electric)[3:end])
        a = _get_val(kernel.a, I, J)
        r01 = _get_val(kernel.r01, I, J)
        t10 = _get_val(kernel.t10, I, J)
        E_val = electric[I, J]
        dE_val = electric_dz[I, J]
        E_plus_ref = E_f[I, J]
        E_minus = 0.5 * (E_val - dE_val / a)
        E_plus = t10 * E_plus_ref + r01 * E_minus
        electric[I,J] = nrm_f * (E_plus + E_minus)
        electric_dz[I,J] = nrm_f * a * (E_plus - E_minus)
    end
end

function apply_boundary_condition_1!(u::HelmholtzField,
                                     E_f::AbstractArray,
                                     p::BidirectionalBPM)
    backend = get_backend(u.electric)
    compute_ft!(p.p_f, u)
    boundary_condition_kernel_1!(backend)(
        u.electric, u.electric_dz, E_f, p.kernel; ndrange = size(u.electric)[1:2])
    compute_ift!(p.p_f, u)
end

@kernel function boundary_condition_kernel_2!(electric, electric_dz, kernel)
    nrm_f = kernel.nrm_f
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(electric)[3:end])
        a = _get_val(kernel.a, I, J)
        r02 = _get_val(kernel.r02, I, J)
        E_val = electric[I, J]
        dE_val = electric_dz[I, J]
        E_plus = 0.5 * (E_val + dE_val / a)
        E_minus = r02 * E_plus
        electric[I,J] = nrm_f * (E_plus + E_minus)
        electric_dz[I,J] = nrm_f * a * (E_plus - E_minus)
    end
end

function apply_boundary_condition_2!(u::HelmholtzField, p::BidirectionalBPM)
    backend = get_backend(u.electric)
    compute_ft!(p.p_f, u)
    boundary_condition_kernel_2!(backend)(
        u.electric, u.electric_dz, p.kernel; ndrange = size(u.electric)[1:2])
    compute_ift!(p.p_f, u)
end

function propagate!(u::HelmholtzField, p::BidirectionalBPM)
    slices_range = 1:size(p.n_xyz, 3)
    for k in reverse(slices_range)
        propagate_slice_backward!(p.ub, p, k)
    end
    kz = compute_kz(u, p.n1)
    E_f = fft(u.electric, (1, 2))
    dEdz_f = fft(u.electric_dz, (1, 2))
    @. E_f = 0.5 * (E_f + dEdz_f  / (im * kz))
    copyto!(u, p.ub)
    apply_boundary_condition_1!(u, E_f, p)
    for k in slices_range
        propagate_slice_forward!(u, p, k)
    end
    apply_boundary_condition_2!(u, p)
    copyto!(p.ub, u)
    u
end
