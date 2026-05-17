struct HelmholtzKernelProp{M, K, T, A} <: AbstractCustomComponent{M}
    z::A
    kz::K
    n0::T
    nrm_f::T

    function HelmholtzKernelProp(u::HelmholtzField{U},
                                 z::Real;
                                 n0::Real = 1) where {T, U <: AbstractArray{Complex{T}}}
        ns = size(u)[1:2]
        A = similar(U, real, 1)
        z_arr = [z] |> A
        kz = compute_kz(u, n0)
        nrm_f = 1/prod(ns)
        K = typeof(kz)
        new{Static, K, T, A}(z_arr, kz, n0, nrm_f)
    end
end

Functors.@functor HelmholtzKernelProp (z,)

_kz_val(kz::AbstractArray{<:Any, 2}, I, J) = kz[I]
function _kz_val(kz::AbstractArray{<:Any, N}, I, J) where {N}
    kz[I, CartesianIndex(min.(Tuple(J), size(kz)[3:end]))]
end

@kernel function helmholtz_propagate_kernel!(electric, electric_dz, kz, z, nrm_f, ::Val{adj}) where {adj}
    I = @index(Global, Cartesian)
    dz = z[1]
    for J in CartesianIndices(axes(electric)[3:end])
        kz_val = _kz_val(kz, I, J)
        E_val = electric[I, J]
        dE_val = electric_dz[I, J]
        
        exp_p = exp(im * kz_val * dz)
        exp_m = conj(exp_p)
        
        C = real((exp_p + exp_m) / 2)
        if iszero(kz_val)
            Ss = typeof(C)(dz)
            Ks = zero(typeof(C))
        else
            diff = exp_p - exp_m
            Ss = real(diff / (2im * kz_val))
            Ks = real(kz_val * diff / 2im)
        end
        
        a, b = adj ? (-Ks, Ss) : (Ss, -Ks)
        electric[I, J] = nrm_f * (C * E_val + a * dE_val)
        electric_dz[I, J] = nrm_f * (b * E_val + C * dE_val)
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

struct HelmholtzProp{M, C} <: AbstractSequence{M}
    optical_components::C

    function HelmholtzProp(optical_components::C) where {N,
                                                          C <: NTuple{N, AbstractPipeComponent}}
        new{Trainable, C}(optical_components)
    end

    function HelmholtzProp(u::HelmholtzField, z::Real; n0::Real = 1.0)
        u_plan = similar(u.electric)
        kernel = HelmholtzKernelProp(u, z; n0)
        wrapper = FourierWrapper(u, kernel, normalize = false)
        M = get_trainability(wrapper)
        optical_components = get_sequence(wrapper)
        C = typeof(optical_components)
        new{M, C}(optical_components)
    end
end

Functors.@functor HelmholtzProp (optical_components,)

get_sequence(p::HelmholtzProp) = p.optical_components
