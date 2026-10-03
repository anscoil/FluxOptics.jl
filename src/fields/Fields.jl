module Fields

using Functors
using AbstractFFTs
using LinearAlgebra
using StaticArrays
using StructArrays
using ..FluxOptics
using ..FluxOptics: isbroadcastable, bzip

import Base: +, -, *, /

export AbstractField, ScalarField, ScalarWaveField
export get_lambdas, get_lambdas_collection
export get_tilts, get_tilts_collection, offset_tilts!
export select_lambdas, select_tilts, set_field_ds!, set_field_data, set_field_tilts
export is_on_axis
export power, normalize_power!, coupling_efficiency, intensity, phase
export orthonormalize, unitary_transform, spatial_moments, spatial_centroids, spatial_variance
export compute_kz, eigen_modes, compute_fresnel, compute_fresnel_r12, compute_fresnel_t12
export Permittivity, ZDecEpsilon, Epsilon
export split_field, poynting_flux, normalize_poynting!
export decompose, recompose, decompose_adjoint, recompose_adjoint
export Direction, Forward, Backward, isforward, isbackward

function parse_val(u::AbstractArray{Complex{T}, N},
                   val::AbstractArray,
                   Nd::Integer) where {N, T}
    shape = ntuple(k -> k <= Nd ? 1 : size(val, k - Nd), N)
    val_adapt = similar(u, T, shape)
    copyto!(val_adapt, val)
    @assert isbroadcastable(val_adapt, u)
    val_adapt
end

function parse_lambdas(u::U, lambdas, Nd::Integer) where {T, U <: AbstractArray{Complex{T}}}
    lambdas_collection = isa(lambdas, Real) ? T(lambdas) : T.(lambdas)
    lambdas_val = isa(lambdas, Real) ? T(lambdas) : parse_val(u, lambdas, Nd)
    (; val = lambdas_val, collection = lambdas_collection)
end

function parse_tilts(u::U, tilts, Nd::Integer) where {T, U <: AbstractArray{Complex{T}}}
    tilts_collection = map(θ -> isa(θ, Real) ? T.([θ]) : T.(θ), tilts)
    tilts_val = map(θ -> parse_val(u, isa(θ, Real) ? [θ] : θ, Nd), tilts)
    (; val = tilts_val, collection = tilts_collection)
end

abstract type AbstractField{U, Nd} end

abstract type Direction end

struct Forward <: Direction end

struct Backward <: Direction end

Base.broadcastable(d::Direction) = Ref(d)

Base.reverse(::Type{Forward}) = Backward
Base.reverse(::Type{Backward}) = Forward
Base.reverse(::Forward) = Backward()
Base.reverse(::Backward) = Forward()
Base.reverse(l, ::Forward) = l
Base.reverse(l, ::Backward) = reverse(l)

Base.sign(::Type{Forward}) = 1
Base.sign(::Type{Backward}) = -1
Base.sign(::Forward) = 1
Base.sign(::Backward) = -1

isforward(::Forward) = true
isforward(::Backward) = false
isbackward(::Forward) = false
isbackward(::Backward) = true

Base.similar(u::AbstractField) = fmap(similar, u)
Base.zero(u::AbstractField) = fmap(zero, u)
Base.copy(u::AbstractField) = fmap(copy, u)

function Base.copyto!(u::AbstractField, v::AbstractField)
    fmap(copyto!, u, v)
    u
end

function rescale!(u::AbstractField, s)
    foreach(a -> a .*= s, Functors.children(u))
    u
end

function normalize_power!(u::AbstractField, target_power = 1;
                          direction::Direction = Forward(), kwargs...)
    P_fwd, P_bwd = power(u; kwargs...)
    rescale!(u, sqrt.(target_power ./ (forward ? P_fwd : P_bwd)))
end

function normalize_poynting!(u::AbstractField, S_out = 1)
    S_in = poynting_flux(u)
    rescale!(u, @. sqrt(abs(S_out / S_in)))
end

include("scalar_field.jl")

include("scalar_wave_field.jl")

include("vectorial_field.jl")

end
