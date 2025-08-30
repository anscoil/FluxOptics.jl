module Fields

using Functors

export ScalarField
export get_data, collect_data
export power, normalize_power!

import Base: +, -, *, /

struct ScalarField{U, Nd, S, T, C}
    data::U
    ds::S
    lambdas::T
    lambdas_collection::C # useful if lambdas is a CuArray

    function ScalarField(u::U, ds::S, lambdas::T,
            lambdas_collection::C) where {U, Nd, S <: NTuple{Nd}, T, C}
        new{U, Nd, S, T, C}(u, ds, lambdas, lambdas_collection)
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
        lambdas = reshape(lambdas, (1, 1, size(u)[(Nd + 1):end]...)) |> U |> real
        lambdas_collection = collect(lambdas)
        new{U, Nd, typeof(ds), typeof(lambdas), typeof(lambdas_collection)}(
            u, ds, lambdas, lambdas_collection)
    end

    function ScalarField(u::U, ds::NTuple{Nd, Real},
            lambda::Real) where {Nd, N, T, U <: AbstractArray{Complex{T}, N}}
        @assert N >= Nd
        new{U, Nd, typeof(ds), T, T}(u, ds, T(lambda), T(lambda))
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

function get_data(u::ScalarField)
    u.data
end

function get_data(u::AbstractArray)
    u
end

function Base.collect(u::ScalarField)
    collect(u.data)
end

function Base.vec(u::ScalarField{U, Nd}) where {U, Nd}
    [ScalarField(data, u.ds, u.lambdas, u.lambdas_collection)
     for data in eachslice(u.data; dims = Tuple((Nd + 1):ndims(u)))]
end

function power(u::AbstractArray{T, N}, ds::NTuple{Nd, Real}) where {T, N, Nd}
    @assert N >= Nd
    dims = ntuple(k -> k, Nd)
    sum(abs2, u; dims = dims) .* prod(ds)
end

function power(u::ScalarField)
    power(u.data, u.ds)
end

function normalize_power!(u::AbstractArray, ds::NTuple, v = 1)
    u .*= sqrt.(v ./ power(u, ds))
    u
end

function normalize_power!(u::ScalarField, v = 1)
    normalize_power!(u.data, u.ds, v)
    u
end

end
