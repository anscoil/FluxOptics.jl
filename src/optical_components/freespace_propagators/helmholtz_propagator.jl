struct HelmholtzKernel{K, T}
    n0::Complex{T}
    a::K
    exp_a_p::K
    exp_a_m::K
    nrm_f::T
end

Adapt.@adapt_structure HelmholtzKernel

function HelmholtzKernel(u::HelmholtzField{U},
                         n0::Number, dz::Real) where {T <: Real,
                                                      U <: AbstractArray{Complex{T}}}
    ns = size(u)[1:2]
    n0 = Complex{T}(n0)
    dz = T(dz)
    a = im * compute_kz(u, n0)
    exp_a_p = @. exp(a * dz)
    exp_a_m = @. exp(-a * dz)
    @. exp_a_m = clamp(abs(exp_a_m), 0, 100) * exp_a_m / abs(exp_a_m)
    nrm_f = T(1/prod(ns))
    HelmholtzKernel(n0, a, exp_a_p, exp_a_m, nrm_f)
end

_get_val(A::AbstractArray{<:Any, 2}, I, J) = A[I]
function _get_val(A::AbstractArray{<:Any, N}, I, J) where {N}
    A[I, CartesianIndex(min.(Tuple(J), size(A)[3:end]))]
end

@kernel function propagate_helmholtz_forward_kernel!(electric, electric_dz, Ep, Em, kernel)
    nrm_f = kernel.nrm_f
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(electric)[3:end])
        a = _get_val(kernel.a, I, J)
        exp_a_p = _get_val(kernel.exp_a_p, I, J)
        exp_a_m = _get_val(kernel.exp_a_m, I, J)
        E_val = electric[I, J]
        dE_val = electric_dz[I, J]
        E_minus = !isnothing(Em) ? Em[I, J] : zero(E_val)
        E_plus = 0.5 * (E_val + dE_val / a)
        E_plus *= exp_a_p
        E_minus *= exp_a_m
        electric[I,J] = nrm_f * (E_plus + E_minus)
        electric_dz[I,J] = nrm_f * a * (E_plus - E_minus)
        if !isnothing(Ep)
            Ep[I, J] = E_plus
        end
    end
end

@kernel function propagate_helmholtz_backward_kernel!(electric, electric_dz, ∂Em, kernel)
    nrm_f = kernel.nrm_f
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(electric)[3:end])
        a = _get_val(kernel.a, I, J)
        exp_a_p = _get_val(kernel.exp_a_p, I, J)
        exp_a_m = _get_val(kernel.exp_a_m, I, J)

        ∂E_val = electric[I, J]
        ∂dE_val = electric_dz[I, J]

        ∂E_plus_prop = nrm_f * (∂E_val + conj(a) * ∂dE_val)
        ∂E_minus_prop = nrm_f * (∂E_val - conj(a) * ∂dE_val)

        if !isnothing(∂Em)
            ∂Em[I, J] = conj(exp_a_m) * ∂E_minus_prop
        end

        ∂E_plus_raw = conj(exp_a_p) * ∂E_plus_prop

        electric[I, J] = 0.5 * ∂E_plus_raw
        electric_dz[I, J] = 0.5 * conj(1 / a) * ∂E_plus_raw
    end
end

struct HelmholtzProp{M, K, E, T, P}  <: AbstractCustomComponent{M}
    trainability::Val{M}
    p_f::P
    E_minus::E
    kernel::HelmholtzKernel{K, T}
    ∂p::Union{Nothing, @NamedTuple{E_minus::E}}
end

trainable(p::HelmholtzProp{<:Trainable}) = (; E_minus = p.E_minus)

get_preallocated_gradient(p::HelmholtzProp{Trainable{Buffered}}) = p.∂p

alloc_saved_buffer(u::HelmholtzField, p::HelmholtzProp{Trainable{Unbuffered}}) = nothing

get_saved_buffer(p::HelmholtzProp{Trainable{Buffered}}) = nothing

function HelmholtzProp(u::HelmholtzField{U}, z::Real;
                       n0::Number = 1,
                       trainable::Bool = false,
                       buffered::Bool = false) where {T <: Real,
                                                      U <: AbstractArray{Complex{T}}}
    M = trainability(trainable, buffered)
    u_plan = similar(u.electric)
    p_f = make_fft_plans(u_plan, (1, 2); normalize = false)
    E_minus = trainable ? similar(u.electric) : nothing
    if !isnothing(E_minus)
        @. E_minus = 0
    end
    kernel = HelmholtzKernel(u, n0, z)
    ∂p = (trainable && buffered) ? (; E_minus = similar(E_minus)) : nothing
    HelmholtzProp(Val(M), p_f, E_minus, kernel, ∂p)
end

function propagate!(u::HelmholtzField, p::HelmholtzProp)
    backend = get_backend(u.electric)
    compute_ft!(p.p_f, u)
    propagate_helmholtz_forward_kernel!(backend)(
        u.electric, u.electric_dz, nothing, p.E_minus, p.kernel;
        ndrange = size(u.electric)[1:2])
    compute_ift!(p.p_f, u)
end

function propagate_and_save!(u::HelmholtzField, ::Nothing, p::HelmholtzProp{<:Trainable})
    propagate!(u, p)
end

function backpropagate_with_gradient!(∂v::HelmholtzField, ::Nothing,
                                      ∂p::NamedTuple, p::HelmholtzProp{<:Trainable})
    backend = get_backend(∂v.electric)
    compute_ft!(p.p_f, ∂v)
    propagate_helmholtz_backward_kernel!(backend)(
        ∂v.electric, ∂v.electric_dz, ∂p.E_minus, p.kernel;
        ndrange = size(∂v.electric)[1:2])
    compute_ift!(p.p_f, ∂v)
    (∂v, ∂p)
end

function backpropagate!(∂v::HelmholtzField, p::HelmholtzProp{Static})
    backend = get_backend(∂v.electric)
    compute_ft!(p.p_f, ∂v)
    propagate_helmholtz_backward_kernel!(backend)(
        ∂v.electric, ∂v.electric_dz, nothing, p.kernel;
        ndrange = size(∂v.electric)[1:2])
    compute_ift!(p.p_f, ∂v)
    ∂v
end
