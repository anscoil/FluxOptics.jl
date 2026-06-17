struct ScalarWaveSource{U, K, T} <: AbstractBidirectionalSource{U}
    u0::ScalarWaveField{U}
    uf::ScalarWaveField{U}
    kz::K
    n0::T
end

Functors.@functor ScalarWaveSource ()

function ScalarWaveSource(u::ScalarWaveField{U};
                          n0::Number = 1.0) where {T, U <: AbstractArray{Complex{T}}}
    n0 = Complex{T}(n0)
    u0 = copy(u)
    uf = similar(u)
    kz = compute_kz(u, n0)
    ScalarWaveSource(u0, uf, kz, n0)
end

Base.size(p::ScalarWaveSource) = size(p.u0)
Base.size(p::ScalarWaveSource, k::Integer) = size(p.u0, k)

get_n0(p::ScalarWaveSource) = p.n0

@kernel function set_forward_field_kernel!(E, dE, Eref, dEref, kz)
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(E)[3:end])
        a = im * _get_val(kz, I, J)
        E1 = 0.5 * (Eref[I,J] + dEref[I,J] / a)
        E2 = 0.5 * (E[I,J] - dE[I,J] / a)
        E[I,J] = E1 + E2
        dE[I,J] =  a * (E1 - E2)
    end
end

@kernel function set_backward_field_kernel!(E, dE, Eref, dEref, kz)
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(E)[3:end])
        a = im * _get_val(kz, I, J)
        E1 = 0.5 * (E[I,J] + dE[I,J] / a)
        E2 = 0.5 * (Eref[I, J] - dEref[I, J] / a)
        E[I,J] = E1 + E2
        dE[I,J] = a * (E1 - E2)
    end
end

function propagate!(u::ScalarWaveField, p::ScalarWaveSource)
    backend = get_backend(u.electric)
    set_forward_field_kernel!(backend)(
        p.u0.electric, p.u0.electric_dz, u.electric, u.electric_dz, p.kz;
        ndrange = size(u.electric)[1:2])
    u
end

function inverse_propagate!(u::ScalarWaveField, p::ScalarWaveSource)
    backend = get_backend(u.electric)
    set_backward_field_kernel!(backend)(
        p.u0.electric, p.u0.electric_dz, u.electric, u.electric_dz, p.kz;
        ndrange = size(u.electric)[1:2])
    u
end

function propagate(p::ScalarWaveSource)
    copyto!(p.uf, p.u0)
    p.uf
end

function Base.fill!(p::ScalarWaveSource, u0::ScalarWaveField)
    copyto!(p.u0, u0)
end

function get_source(p::ScalarWaveSource)
    p.u0
end
