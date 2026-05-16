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
        # C = real(cos(kz_val * dz))
        C = cos(kz_val * dz)
        sin_kz_dz = sin(kz_val * dz)
        # Ss = iszero(kz_val) ? typeof(C)(dz) : real(sin_kz_dz / kz_val)
        # Ks = real(kz_val * sin_kz_dz)
        Ss = iszero(kz_val) ? typeof(C)(dz) : sin_kz_dz / kz_val
        Ks = kz_val * sin_kz_dz
        a, b = adj ? (-Ks, Ss) : (conj(Ss), -conj(Ks))
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
