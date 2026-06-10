struct BidirectionalBPM{M, N, K, E, T, P}  <: AbstractCustomComponent{M}
    trainability::Val{M}
    n_xyz::N
    n0::Complex{T}
    dz::T
    p_f::P
    E_plus::Union{Nothing, E}
    E_minus::E
    kernel::HelmholtzKernel{K, T}
    ∂p::Union{Nothing, @NamedTuple{E_minus::E}}
end

Functors.@functor BidirectionalBPM (n_xyz, E_minus)

trainable(p::BidirectionalBPM{<:Trainable}) = (; E_minus = p.E_minus)

get_preallocated_gradient(p::BidirectionalBPM{Trainable{Buffered}}) = p.∂p

function alloc_saved_buffer(u::HelmholtzField, p::BidirectionalBPM{Trainable{Unbuffered}})
    n_slices = size(p.n_xyz, 3)
    similar(u.electric, (size(u.electric)..., n_slices))
end

get_saved_buffer(p::BidirectionalBPM{Trainable{Buffered}}) = p.E_plus

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
                          trainable::Bool = false,
                          buffered::Bool = false) where {T <: Real,
                                                         U <: AbstractArray{Complex{T}}}
    ns = size(u)[1:2]
    n_slices = size(n_xyz, 3)
    @assert size(n_xyz)[1:2] == ns
    @assert n_slices >= 2
    dz = T(thickness / n_slices)
    M = trainability(trainable, buffered)
    N = isreal(n_xyz) ? T : Complex{T}
    n_xyz_buf = similar(u.electric, N, size(n_xyz))
    copyto!(n_xyz_buf, n_xyz)
    n0 = Complex{T}(n0)
    u_plan = similar(u.electric)
    p_f = make_fft_plans(u_plan, (1, 2); normalize = false)
    E_plus = (trainable && buffered) ? similar(u.electric,
                                               (size(u.electric)..., n_slices)) : nothing
    E_minus = trainable ? similar(u.electric, (size(u.electric)..., n_slices)) : nothing
    if !isnothing(E_plus)
        @. E_plus = 0
    end
    if !isnothing(E_minus)
        @. E_minus = 0
    end
    kernel = HelmholtzKernel(u, n0, dz)
    ∂p = (trainable && buffered) ? (; E_minus = similar(E_minus)) : nothing
    BidirectionalBPM(Val(M), n_xyz_buf, n0, dz, p_f, E_plus, E_minus, kernel, ∂p)
end

function propagate_slice_forward!(u::HelmholtzField, p::BidirectionalBPM,
                                  Ep::Union{Nothing, AbstractArray}, k::Integer)
    backend = get_backend(u.electric)
    n_xy = view(p.n_xyz, :, :, k)
    compute_ft!(p.p_f, u)
    E_plus = !isnothing(Ep) ? selectdim(Ep, ndims(Ep), k) : nothing
    E_minus = !isnothing(p.E_minus) ? selectdim(p.E_minus, ndims(p.E_minus), k) : nothing
    propagate_helmholtz_forward_kernel!(backend)(
        u.electric, u.electric_dz, E_plus, E_minus, p.kernel;
        ndrange = size(u.electric)[1:2])
    compute_ift!(p.p_f, u)
    @. u.electric_dz += ((2π/u.lambdas.val)^2 * (p.n0^2 - n_xy^2) * p.dz * u.electric)
end

function propagate_slice_backward!(u::HelmholtzField, p::BidirectionalBPM,
                                   ∂Em::AbstractArray, k::Integer)
    backend = get_backend(u.electric)
    n_xy = view(p.n_xyz, :, :, k)
    @. u.electric += ((2π/u.lambdas.val)^2 * (p.n0^2 - n_xy^2) * p.dz * u.electric_dz)
    compute_ft!(p.p_f, u)
    ∂E_minus = selectdim(∂Em, ndims(∂Em), k)
    propagate_helmholtz_backward_kernel!(backend)(
        u.electric, u.electric_dz, ∂E_minus, p.kernel;
        ndrange = size(u.electric)[1:2])
    compute_ift!(p.p_f, u)
end

@kernel function boundary_condition_kernel_1!(electric, electric_dz, E_m,
                                              r01_arr, t10_arr, kz_n1, kernel)
    nrm_f = kernel.nrm_f
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(electric)[3:end])
        a = _get_val(kernel.a, I, J)
        r01 = _get_val(r01_arr, I, J)
        t10 = _get_val(t10_arr, I, J)
        kz1 = _get_val(kz_n1, I, J)
        E_val = electric[I, J]
        dE_val = electric_dz[I, J]
        E_plus_ref = 0.5 * (E_val + dE_val / (im * kz1))
        E_minus = _get_val(E_m, I, J)
        E_plus = t10 * E_plus_ref + r01 * E_minus
        electric[I, J] = nrm_f * (E_plus + E_minus)
        electric_dz[I, J] = nrm_f * a * (E_plus - E_minus)
    end
end

function apply_boundary_condition_1!(u::HelmholtzField,
                                     E_m::AbstractArray,
                                     p::BidirectionalBPM)
    n1 = 1.5
    backend = get_backend(u.electric)
    r01 = compute_fresnel_r12(u, p.n0, n1)
    t10 = compute_fresnel_t12(u, n1, p.n0)
    kz_n1 = compute_kz(u, n1)    
    compute_ft!(p.p_f, u)
    boundary_condition_kernel_1!(backend)(
        u.electric, u.electric_dz, E_m, r01, t10, kz_n1, p.kernel;
        ndrange = size(u.electric)[1:2])
    compute_ift!(p.p_f, u)
end

@kernel function boundary_condition_kernel_1_backward!(electric, electric_dz,
                                                       ∂E_m, r01_arr, t10_arr,
                                                       kz_n1, kernel)
    nrm_f = kernel.nrm_f
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(electric)[3:end])
        a = _get_val(kernel.a, I, J)
        r01 = _get_val(r01_arr, I, J)
        t10 = _get_val(t10_arr, I, J)
        kz1 = _get_val(kz_n1, I, J)

        g_E = electric[I, J]
        g_dE = electric_dz[I, J]

        ∂E_plus = nrm_f * (g_E + conj(a) * g_dE)
        ∂E_minus = nrm_f * (g_E - conj(a) * g_dE) + conj(r01) * ∂E_plus
        ∂E_plus_ref = conj(t10) * ∂E_plus

        if !isnothing(∂E_m)
            ∂E_m[I, J] += ∂E_minus
        end

        electric[I, J] = 0.5 * ∂E_plus_ref
        electric_dz[I, J] = 0.5 * conj(one(kz1) / (im * kz1)) * ∂E_plus_ref
    end
end

function apply_boundary_condition_1_backward!(u::HelmholtzField, p::BidirectionalBPM, E_m)
    n1 = 1.5
    backend = get_backend(u.electric)
    r01 = compute_fresnel_r12(u, p.n0, n1)
    t10 = compute_fresnel_t12(u, n1, p.n0)
    kz_n1 = compute_kz(u, n1)
    compute_ft!(p.p_f, u)
    boundary_condition_kernel_1_backward!(backend)(
        u.electric, u.electric_dz, E_m, r01, t10, kz_n1, p.kernel;
        ndrange = size(u.electric)[1:2])
    compute_ift!(p.p_f, u)
end

function _propagate!(u::HelmholtzField, p::BidirectionalBPM; Ep=nothing)
    slices_range = 1:size(p.n_xyz, 3)
    for k in slices_range
        propagate_slice_forward!(u, p, Ep, k)
    end
    u
end

propagate!(u::HelmholtzField, p::BidirectionalBPM) = _propagate!(u, p)

function propagate_and_save!(u::HelmholtzField,
                             Ep::AbstractArray,
                             p::BidirectionalBPM{<:Trainable})
    E_m = selectdim(p.E_minus, ndims(p.E_minus), 1)
    apply_boundary_condition_1!(u, E_m, p)
    _propagate!(u, p; Ep)
end

function backpropagate_with_gradient!(∂v::HelmholtzField,
                                      Ep::AbstractArray,
                                      ∂p::NamedTuple,
                                      p::BidirectionalBPM{<:Trainable})
    slices_range = 1:size(p.n_xyz, 3)
    for k in reverse(slices_range)
        propagate_slice_backward!(∂v, p, ∂p.E_minus, k)
    end
    E_m = selectdim(∂p.E_minus, ndims(∂p.E_minus), 1)
    apply_boundary_condition_1_backward!(∂v, p, E_m)
    (∂v, ∂p)
end
