module Fields

using Functors
using LinearAlgebra
using ..FluxOptics
using ..FluxOptics: isbroadcastable, bzip

import Base: +, -, *, /

export ScalarField
export get_lambdas, get_lambdas_collection
export get_tilts, get_tilts_collection
export select_lambdas, select_tilts, set_field_ds, set_field_data, set_field_tilts
export power, normalize_power!

function parse_tilt_vectors(u::U,
        θs::NTuple{Nd, Union{Real, AbstractVector{<:Real}}}
) where {Nd, U <: AbstractArray{<:Complex}}
    shape = ntuple(k -> k <= Nd ? 1 : size(u, k), ndims(u))
    V = adapt_dim(U, 1, real)
    map(θ -> isa(θ, Real) ? V([θ]) : reshape(V(θ), shape), θs)
end

function collect_tilt_vectors(u::U,
        θs::NTuple{Nd, Union{Real, AbstractVector{<:Real}}}
) where {Nd, T <: Real, U <: AbstractArray{Complex{T}}}
    shape = ntuple(k -> k <= Nd ? 1 : size(u, k), ndims(u))
    map(θ -> isa(θ, Real) ? fill(T(θ), shape) : reshape(Array{T}(θ), shape), θs)
end

function parse_val(u::U, val::AbstractArray,
        Nd::Integer
) where {N, T, U <: AbstractArray{Complex{T}, N}}
    shape = ntuple(k -> k <= Nd ? 1 : size(val, k - Nd), N)
    val = reshape(val, shape) |> U |> real
    @assert isbroadcastable(val, u)
    val
end

function parse_lambdas(u::U, lambdas, Nd::Integer) where {T, U <: AbstractArray{Complex{T}}}
    lambdas_collection = isa(lambdas, Real) ? T(lambdas) : T.(lambdas)
    lambdas_val = isa(lambdas, Real) ? T(lambdas) : parse_val(u, lambdas, Nd)
    (; val = lambdas_val, collection = lambdas_collection)
end

function parse_tilts(u::U, tilts, Nd::Integer) where {T, U <: AbstractArray{Complex{T}}}
    tilts_collection = map(θ -> isa(θ, Real) ? T(θ) : T.(θ), tilts)
    tilts_val = map(θ -> isa(θ, Real) ? T(θ) : parse_val(u, θ, Nd), tilts)
    (; val = tilts_val, collection = tilts_collection)
end

struct ScalarField{U, Nd, S, L, A}
    data::U
    ds::S
    lambdas::L
    tilts::A

    function ScalarField(u::U, ds::S, lambdas::L,
            tilts::A
    ) where {U, Nd, S <: NTuple{Nd}, L <: NamedTuple, A <: NamedTuple}
        new{U, Nd, S, L, A}(u, ds, lambdas, tilts)
    end

    function ScalarField(u::U, ds::S,
            lambdas::Union{Real, AbstractArray{<:Real}};
            tilts::NTuple{Nd, Union{<:Real, <:AbstractArray}} = ntuple(_ -> 0, Nd)
    ) where {Nd, N, S <: NTuple{Nd, Real}, T, U <: AbstractArray{Complex{T}, N}}
        @assert N >= Nd
        lambdas = parse_lambdas(u, lambdas, Nd)
        tilts = parse_tilts(u, tilts, Nd)
        L = typeof(lambdas)
        A = typeof(tilts)
        new{U, Nd, S, L, A}(u, ds, lambdas, tilts)
    end

    function ScalarField(
            nd::NTuple{N, Integer}, ds::NTuple{Nd, Real}, lambdas;
            tilts = ntuple(_ -> 0, Nd)) where {N, Nd}
        u = zeros(ComplexF64, nd)
        ScalarField(u, ds, lambdas; tilts)
    end
end

Functors.@functor ScalarField (data,)

function get_lambdas(u::ScalarField)
    u.lambdas.val
end

function get_lambdas_collection(u::ScalarField)
    u.lambdas.collection
end

function select_lambdas(u::ScalarField)
    function select(is_collection::Bool)
        is_collection ? u.lambdas.collection : u.lambdas.val
    end
    select
end

function get_tilts(u::ScalarField)
    u.tilts.val
end

function get_tilts_collection(u::ScalarField)
    u.tilts.collection
end

function select_tilts(u::ScalarField)
    Tuple([is_collection -> is_collection ? collection : val
           for (collection, val) in zip(u.tilts.collection, u.tilts.val)])
end

function set_field_ds(u::ScalarField{U, Nd}, ds::NTuple{Nd, Real}) where {U, Nd}
    ScalarField(u.data, ds, u.lambdas, u.tilts)
end

function set_field_data(u::ScalarField{U, Nd}, data::V) where {U, V, Nd}
    ScalarField(data, u.ds, u.lambdas.collection; tilts = u.tilts.collection)
end

function set_field_data(u::ScalarField{U, Nd}, data::U) where {U, Nd}
    ScalarField(data, u.ds, u.lambdas, u.tilts)
end

function set_field_tilts(u::ScalarField{U, Nd}, tilts) where {U, Nd}
    ScalarField(u.data, u.ds, u.lambdas.collection; tilts)
end

# function Base.broadcastable(sf::ScalarField)
#     return Ref(sf)
# end

function Base.broadcasted(f, u::ScalarField)
    ScalarField(complex(broadcast(f, u.data)), u.ds, u.lambdas.collection;
        tilts = u.tilts.collection)
end

# function Base.broadcasted(f, a::ScalarField, b::AbstractArray)
#     ScalarField(broadcast(f, a.data, b), a.lambdas)
# end

function +(u::ScalarField, v::ScalarField)
    set_field_data(u, u.data + v.data)
end

Base.getindex(u::ScalarField, i...) = view(u.data, i...)
Base.size(u::ScalarField) = size(u.data)
Base.size(u::ScalarField, k::Integer) = size(u.data, k)

function Base.ndims(u::ScalarField{U, Nd}, spatial::Bool = false) where {U, Nd}
    spatial ? Nd : ndims(u.data)
end

Base.eltype(u::ScalarField) = eltype(u.data)

function Base.fill!(u::ScalarField, v)
    u.data .= v
    u
end

function Base.fill!(u::ScalarField, v::AbstractArray)
    copyto!(u.data, v)
    u
end

function Base.copy(u::ScalarField)
    set_field_data(u, copy(u.data))
end

function Base.copyto!(u::ScalarField, v::ScalarField)
    copyto!(u.data, v.data)
    u
end

function Base.similar(u::ScalarField)
    set_field_data(u, similar(u.data))
end

function Base.collect(u::ScalarField)
    collect(u.data)
end

function Base.vec(u::ScalarField{U, Nd}) where {U, Nd}
    u_slices = eachslice(u.data; dims = Tuple((Nd + 1):ndims(u)))
    [ScalarField(data, u.ds, lambda; tilts)
     for (data, lambda, tilts...) in
         bzip(u_slices, u.lambdas.collection, u.tilts.collection...)]
end

function FluxOptics.intensity(u::ScalarField{U, Nd}) where {U, Nd}
    reshape(sum(intensity, u.data; dims = Tuple((Nd + 1):ndims(u))), size(u)[1:Nd])
end

function FluxOptics.correlation(u::ScalarField{U, Nd},
        v::ScalarField{V, Nd}) where {U, V, Nd}
    u_vec = vec(u)
    v_vec = vec(v)
    [correlation(u.data, v.data) for (u, v) in zip(u_vec, v_vec)]
end

function LinearAlgebra.dot(u::ScalarField{U, Nd},
        v::ScalarField{V, Nd}) where {U, V, Nd}
    u_vec = vec(u)
    v_vec = vec(v)
    [dot(u.data, v.data) for (u, v) in zip(u_vec, v_vec)]
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
    set_field_data(u, conj(u.data))
end

end
