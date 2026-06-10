struct HelmholtzKernelProp{M, K, T, A} <: AbstractCustomComponent{M}
    trainability::Val{M}
    z::A
    kz::K
    n0::Complex{T}
    nrm_f::T
end

function HelmholtzKernelProp(u::HelmholtzField{U},
                             z::Real;
                             n0::Number = 1.0) where {T, U <: AbstractArray{Complex{T}}}
    ns = size(u)[1:2]
    A = similar(U, real, 1)
    z_arr = [z] |> A
    n0 = Complex{T}(n0)
    kz = compute_kz(u, n0)
    nrm_f = T(1/prod(ns))
    HelmholtzKernelProp(Val(Static), z_arr, kz, n0, nrm_f)
end

Functors.@functor HelmholtzKernelProp (z,)

_kz_val(kz::AbstractArray{<:Any, 2}, I, J) = kz[I]
function _kz_val(kz::AbstractArray{<:Any, N}, I, J) where {N}
    kz[I, CartesianIndex(min.(Tuple(J), size(kz)[3:end]))]
end

function bounded_exp(E::Complex{T}, exp_m::Complex{T}) where T
    sat_val = T(1)
    E_new = E * exp_m
    amp = abs(E_new)
    E_new * sat_val / (amp + sat_val)
end

@kernel function helmholtz_propagate_kernel!(electric, electric_dz, kz, z, nrm_f, ::Val{adj}) where {adj}
    I = @index(Global, Cartesian)
    dz = z[1]
    T = typeof(dz)
    for J in CartesianIndices(axes(electric)[3:end])
        kz_val = _kz_val(kz, I, J)
        a = im * kz_val
        exp_p = exp(a * dz)
        # exp_m = conj(exp_p)
        exp_m = exp(-a * dz)
        
        E_val = electric[I, J]
        dE_val = electric_dz[I, J]

        E_minus = 0
        if adj
            E_plus = nrm_f * (E_val + conj(a) * dE_val)
            E_minus = 2 * nrm_f * E_val - E_plus
            E_plus *= conj(exp_p)
            # E_minus *= conj(exp_m)
            cm = conj(exp_m)
            E_minus *= cm / max(one(real(cm)), abs(cm))
            electric[I,J] = T(0.5) * (E_plus + E_minus)
            electric_dz[I,J] = T(0.5) / conj(a) * (E_plus - E_minus)
        else
            E_minus = T(0.5) * (E_val - dE_val / a)
            E_plus = T(0.5) * (E_val + dE_val / a)
            E_plus *= exp_p
            # E_minus *= exp_m
            E_minus = bounded_exp(E_minus, exp_m)
            electric[I,J] = nrm_f * (E_plus + E_minus)
            electric_dz[I,J] = nrm_f * a * (E_plus - E_minus)
        end
    end
end

function propagate!(u::HelmholtzField, p::HelmholtzKernelProp)
    backend = get_backend(u.electric)
    helmholtz_propagate_kernel!(backend)(
        u.electric, u.electric_dz, p.kz, p.z, p.nrm_f, Val(false);
        ndrange = size(u.electric)[1:2])
    u
end

function backpropagate!(u::HelmholtzField, p::HelmholtzKernelProp)
    backend = get_backend(u.electric)
    helmholtz_propagate_kernel!(backend)(
        u.electric, u.electric_dz, p.kz, p.z, p.nrm_f, Val(true);
        ndrange = size(u.electric)[1:2])
    u
end

struct HelmholtzBoundedProp{M, C} <: AbstractSequence{M}
    optical_components::C

    function HelmholtzBoundedProp(optical_components::C
                                  ) where {N, C <: NTuple{N, AbstractPipeComponent}}
        new{Trainable, C}(optical_components)
    end

    function HelmholtzBoundedProp(u::HelmholtzField, z::Real; n0::Number = 1.0)
        u_plan = similar(u.electric)
        kernel = HelmholtzKernelProp(u, z; n0)
        wrapper = FourierWrapper(u, kernel, normalize = false)
        M = get_trainability(wrapper)
        optical_components = get_sequence(wrapper)
        C = typeof(optical_components)
        new{M, C}(optical_components)
    end
end

Functors.@functor HelmholtzBoundedProp (optical_components,)

get_sequence(p::HelmholtzBoundedProp) = p.optical_components
