struct ScalarWaveSource{U, K, T}
    u0::U
    uf::U
    kz::K
    n0::T
end

Functors.@functor ScalarWaveSource ()

function HelmholtzSource(u::ScalarWaveField{U};
                         n0::Number = 1.0) where {T, U <: AbstractArray{Complex{T}}}
    n0 = Complex{T}(n0)
    u0 = copy(u)
    uf = similar(u)
    kz = compute_kz(u, n0)
    ScalarWaveSource(u0, uf, kz, n0)
end

Base.size(p::ScalarWaveSource) = size(p.u0)
Base.size(p::ScalarWaveSource, k::Integer) = size(p.u0, k)

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
