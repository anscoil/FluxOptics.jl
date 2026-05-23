struct BidirectionalKernel{K, T}
    a::K
    exp_a_p::K
    exp_a_m::K
    nrm_f::T
end

Adapt.@adapt_structure BidirectionalKernel

struct BidirectionalBPM{M, N, K, E, T, P}  <: AbstractCustomComponent{M}
    trainability::Val{M}
    n_xyz::N
    n0::Complex{T}
    dz::T
    p_f::P
    E_tmp::E
    β::Ref{T}
    kernel::BidirectionalKernel{K, T}
end

Functors.@functor BidirectionalBPM (n_xyz,)

function optimal_gauge(n_xyz::AbstractArray)
    re_n2 = real.(n_xyz.^2)
    n0 = sqrt(maximum(re_n2))
    ϵ = sqrt(n0^2 - minimum(re_n2))
    return complex(n0, ϵ)
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
    M  = trainability(trainable, buffered)
    N = isreal(n_xyz) ? T : Complex{T}
    n_xyz_buf = similar(u.electric, N, size(n_xyz))
    copyto!(n_xyz_buf, n_xyz)
    n0 = Complex{T}(n0)
    a = im * compute_kz(u, n0)
    exp_a_p = @. exp(a * dz)
    exp_a_m = @. exp(-a * dz)
    u_plan = similar(u.electric)
    p_f = make_fft_plans(u_plan, (1, 2); normalize = false)
    E_tmp = similar(u.electric, (size(u.electric)..., n_slices))
    @. E_tmp = 0
    nrm_f = T(1/prod(ns))
    kernel = BidirectionalKernel(a, exp_a_p, exp_a_m, nrm_f)
    BidirectionalBPM(Val(M), n_xyz_buf, n0, dz, p_f, E_tmp, Ref(T(0)), kernel)
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

function propagate_slice_forward!(u::HelmholtzField, p::BidirectionalBPM, k::Integer)
    β = p.β[]
    @assert 0 <= β <= 1
    backend = get_backend(u.electric)
    n_xy = view(p.n_xyz, :, :, k)
    @. u.electric_dz += ((2π/u.lambdas.val)^2 * (p.n0^2 - n_xy^2) * p.dz * u.electric)
    compute_ft!(p.p_f, u)
    E_minus = selectdim(p.E_tmp, ndims(p.E_tmp), k)
    propagate_helmholtz_forward_kernel!(backend)(
        u.electric, u.electric_dz, E_minus, p.kernel, β; ndrange = size(u.electric)[1:2])
    compute_ift!(p.p_f, u)
end

function propagate!(u::HelmholtzField, p::BidirectionalBPM)
    for k in 1:size(p.n_xyz, 3)
        propagate_slice_forward!(u, p, k)
    end
    u
end
