module Fields

using Functors
using ..FluxOptics

export ScalarField
export power, normalize_power!

import Base: +, -, *, /

struct ScalarField{U, Nd, T, S, C}
    data::U
    ds::S
    lambdas::T
    lambdas_collection::C # useful if lambdas is a CuArray

    function ScalarField(u::U, ds::S, lambdas::T,
            lambdas_collection::C) where {U, Nd, S <: NTuple{Nd}, T, C}
        new{U, Nd, T, S, C}(u, ds, lambdas, lambdas_collection)
    end

    function ScalarField(u::U,
            ds::NTuple{Nd, Real},
            lambdas::T) where {
            Nd, N, U <: AbstractArray{<:Complex, N},
            T <: AbstractArray{<:Real}}
        @assert N >= Nd
        ns = size(u)[1:Nd]
        nr = div(length(u), prod(ns))
        @assert length(lambdas) == nr
        lambdas = reshape(lambdas, ntuple(k -> k <= Nd ? 1 : size(u, k), N)) |> U |> real
        lambdas_collection = collect(lambdas)
        new{U, Nd, typeof(lambdas), typeof(ds), typeof(lambdas_collection)}(
            u, ds, lambdas, lambdas_collection)
    end

    function ScalarField(u::U, ds::NTuple{Nd, Real},
            lambda::Real) where {Nd, N, T, U <: AbstractArray{Complex{T}, N}}
        @assert N >= Nd
        new{U, Nd, T, typeof(ds), T}(u, ds, T(lambda), T(lambda))
    end

    function ScalarField(
            nd::NTuple{N, Integer}, ds::NTuple{Nd, Real}, lambdas) where {N, Nd}
        u = zeros(ComplexF64, nd)
        ScalarField(u, ds, lambdas)
    end
end

Functors.@functor ScalarField (data,)

function Base.broadcastable(sf::ScalarField)
    return Ref(sf)
end

function Base.broadcasted(f, a::ScalarField, b::AbstractArray)
    ScalarField(broadcast(f, a.data, b), a.lambdas)
end

function +(a::ScalarField, b::ScalarField)
    ScalarField(a.data + b.data, a.ds, a.lambdas)
end

Base.getindex(u::ScalarField, i...) = view(u.data, i...)
Base.size(u::ScalarField) = size(u.data)
Base.ndims(u::ScalarField) = ndims(u.data)

function Base.fill!(u::ScalarField, v)
    u.data .= v
    u
end

function Base.fill!(u::ScalarField, v::AbstractArray)
    copyto!(u.data, v)
    u
end

function Base.copy(u::ScalarField)
    ScalarField(copy(u.data), u.ds, u.lambdas, u.lambdas_collection)
end

function Base.copyto!(u::ScalarField, v::ScalarField)
    copyto!(u.data, v.data)
    u
end

function Base.similar(u::ScalarField)
    ScalarField(similar(u.data), u.ds, u.lambdas, u.lambdas_collection)
end

function Base.collect(u::ScalarField)
    collect(u.data)
end

function Base.vec(u::ScalarField{U, Nd}) where {U, Nd}
    [ScalarField(data, u.ds, u.lambdas, u.lambdas_collection)
     for data in reshape(eachslice(u.data; dims = Tuple((Nd + 1):ndims(u))), :)]
end

function FluxOptics.intensity(u::ScalarField{U, Nd}) where {U, Nd}
    reshape(sum(intensity, u.data; dims = Tuple((Nd + 1):ndims(u))), size(u)[1:Nd])
end

function FluxOptics.correlation(u::ScalarField{U, Nd}, v::ScalarField{U, Nd}) where {U, Nd}
    u_vec = vec(u)
    v_vec = vec(v)
    [correlation(u.data, v.data) for (u, v) in zip(u_vec, v_vec)]
end

function power(u::ScalarField{U, Nd}) where {U, Nd}
    dims = ntuple(k -> k, Nd)
    sum(abs2, u.data; dims = dims) .* prod(u.ds)
end

function normalize_power!(u::ScalarField, v = 1)
    u.data .*= sqrt.(v ./ power(u))
    u
end

function Base.conj(u::ScalarField)
    ScalarField(conj(u.data), u.ds, u.lambdas, u.lambdas_collection)
end

end
