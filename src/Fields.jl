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

"""
    ScalarField(data::AbstractArray{Complex}, ds::NTuple{Nd,Real}, lambdas; tilts=ntuple(_->0, Nd))

Represent scalar optical fields with spatial grid information, wavelength(s), and optional tilt angles.

This is the central data structure of FluxOptics.jl for storing and manipulating scalar optical fields.
The fields data can be multi-dimensional with 1 or 2 transverse dimensions and additional dimensions for 
different spatial modes. Each mode can hold independent wavelength and tilt information.

# Arguments
- `data::AbstractArray{Complex}`: Complex field amplitude data.
- `ds::NTuple{Nd,Real}`: Spatial sampling intervals (dx,[dy]) in meters.
- `lambdas`: Wavelength(s) - can be a scalar Real or AbstractArray{Real} for multiple wavelengths.
  In case of AbstractArray{Real}, it must be broadcastable on the extra dimensions of `data`
  (all dimensions except the spatial ones).
- `tilts::NTuple{Nd}=ntuple(_->0, Nd)`: Tilt angles in radians corresponding to a Fourier offset
  of fx₀ = sin(θx)/λ, fy₀ = sin(θy)/λ.
  Array arguments to NTuple are accepted as long as they broadcast with the extra non-spatial
  dimensions of `data`.

# Examples
```julia
# Simple 2D field at 1064nm with 1μm pixel size
data = zeros(ComplexF64, 256, 256)
u = ScalarField(data, (1e-6, 1e-6), 1064e-9)

# Multi-wavelength field
wavelengths = [800e-9, 1064e-9, 1550e-9]
data = zeros(ComplexF64, 256, 256, 3)
u = ScalarField(data, (1e-6, 1e-6), wavelengths)

# Field with initial tilt
u = ScalarField(data, (1e-6, 1e-6), 1064e-9; tilts=(0.01, 0.005))
```

See also: [`power`](@ref), [`normalize_power!`](@ref)
"""
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
        @assert Nd in (1, 2)
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

"""
    set_field_data(u::ScalarField, data::AbstractArray)

Create a new ScalarField with updated field data, preserving all other parameters.

This function creates a copy with new amplitude data while keeping the same spatial grid,
wavelengths, and tilt information.

# Arguments
- `u::ScalarField`: Original field
- `data::AbstractArray`: New complex field data

# Returns
New `ScalarField` with updated data

# Examples
```julia
u = ScalarField(zeros(ComplexF64, 256, 256), (1e-6, 1e-6), 1064e-9)
new_data = rand(ComplexF64, 256, 256)
u_new = set_field_data(u, new_data)
```
"""
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

"""
    copy(u::ScalarField)

Create a copy of the scalar field.

Creates a copy of the field data, while sharing the internal representation of other parameters.

# Examples
```julia
u = ScalarField(data, (1e-6, 1e-6), 1064e-9)
u_copy = copy(u)
# Modifying u_copy.data will not affect u, but the internal arrays representing 
# the wavelengths or tilts must never be modified.
```

See also: [`similar`](@ref)
"""
function Base.copy(u::ScalarField)
    set_field_data(u, copy(u.data))
end

function Base.copyto!(u::ScalarField, v::ScalarField)
    copyto!(u.data, v.data)
    u
end

"""
    similar(u::ScalarField)

Create a new ScalarField with same parameters but uninitialized data.

Useful for creating temporary fields with the same grid, wavelengths and tilts
structure as an existing field.

# Examples
```julia
u = ScalarField(data, (1e-6, 1e-6), 1064e-9)
u_temp = similar(u)  # Same grid/wavelengths/tilts, but data is uninitialized
```

See also: [`copy`](@ref), [`set_field_data`](@ref)
"""
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

"""
    power(u::ScalarField)

Compute the optical power of the field.

Calculates the spatial integral of the intensity |u|² over the field domain,
properly accounting for the spatial sampling.

# Mathematical definition
P = ∫∫ |u(x[,y])|² dx [dy] ≈ Σᵢⱼ |u[i[,j]]|² × dx [× dy].

# Returns  
Array of power value(s) with same dimensions as `u`, spatial dimensions being reduced to size 1.

# Examples
```julia
data = rand(ComplexF64, 256, 256, 3)
u = ScalarField(data, (1e-6, 1e-6), 1064e-9)
P = power(u)  # Returns 1×1×3 Array

```

See also: [`normalize_power!`](@ref)
"""
function power(u::ScalarField{U, Nd}) where {U, Nd}
    dims = ntuple(k -> k, Nd)
    sum(abs2, u.data; dims = dims) .* prod(u.ds)
end

"""
    normalize_power!(u::ScalarField, target_power=1)

Normalize the field to have specified optical power (in-place modification).

Scales the field amplitudes so that the total optical power equals the target values.
This is useful for setting consistent power levels between different fields.

# Arguments
- `u::ScalarField`: Field to normalize (modified in-place)
- `target_power=1`: Target power value(s) - can be a scalar Real or AbstractArray{Real} as long as it
  can be broadcasted on the field data.

# Returns
The modified field `u`

# Examples
```julia
u = ScalarField(rand(ComplexF64, 256, 256, 3), (1e-6, 1e-6), 1064e-9)
normalize_power!(u)        # Normalize all fields to 1 W
normalize_power!(u, 1e-3)  # Normalize all fields to 1 mW

# For multiple fields: normalize each field separately
normalize_power!(u, reshape([1e-3, 2e-3, 1e-3], 1, 1, 3))  # Different power per field
```

See also: [`power`](@ref)
"""
function normalize_power!(u::ScalarField, v = 1)
    u.data .*= sqrt.(v ./ power(u))
    u
end

function Base.conj(u::ScalarField)
    set_field_data(u, conj(u.data))
end

end
