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
export power, normalize_power!, coupling_efficiency, intensity, phase

function parse_val(u::U,
                   val::AbstractArray,
                   Nd::Integer) where {N, T, U <: AbstractArray{Complex{T}, N}}
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
    ScalarField(nd::NTuple, ds::NTuple{Nd,Real}, lambdas; tilts=ntuple(_->0, Nd))

Represent a scalar optical field with spatial grid information, wavelength(s), and optional tilt angles.

This is the central data structure of FluxOptics.jl for storing and manipulating scalar optical fields.
The field data can be multi-dimensional with 1 or 2 transverse dimensions and additional dimensions for 
different spatial modes. Each mode can hold independent wavelength and tilt information.

# Constructors

**From existing data:**
- `data::AbstractArray{Complex}`: Complex field amplitude data.
- `ds::NTuple{Nd,Real}`: Spatial sampling intervals (dx,[dy]) in micrometers (or meters as long as
  consistent units are used everywhere).
- `lambdas`: Wavelength(s) - can be a scalar Real or AbstractArray{Real} for multiple wavelengths.
  In case of AbstractArray{Real}, it must be broadcastable on the extra dimensions of `data`
  (all dimensions except the spatial ones).
- `tilts::NTuple{Nd}=ntuple(_->0, Nd)`: Tilt angles in radians corresponding to a Fourier offset
  of fx₀ = sin(θx)/λ, fy₀ = sin(θy)/λ.
  Array arguments to NTuple are accepted as long as they broadcast with the extra non-spatial
  dimensions of `data`.

**Zero-initialized field (convenience):**
- `nd::NTuple`: Dimensions of the data array (nx, [ny,] ...).
- Other arguments same as above.

# Examples

**Creating from existing data:**
```jldoctest
# Small example for documentation - use larger arrays in practice
julia> data = rand(ComplexF64, 4, 4);

julia> u = ScalarField(data, (1.0, 1.0), 1.064);

julia> size(u)
(4, 4)
```

**Creating zero-initialized field:**
```jldoctest
# Small example for documentation - use larger arrays in practice
julia> u = ScalarField((4, 4), (1.0, 1.0), 1.064);

julia> size(u)
(4, 4)
```

**Multi-wavelength field:**
```jldoctest
julia> wavelengths = [0.8, 1.064, 1.550];

# Small example for documentation - use larger arrays in practice
julia> data = zeros(ComplexF64, 4, 4, 3);

julia> u = ScalarField(data, (1.0, 1.0), wavelengths);
```

**Field with initial tilt:**
```jldoctest
julia> wavelengths = [0.8, 1.064, 1.55];

# Small example for documentation - use larger arrays in practice
julia> data = zeros(ComplexF64, 4, 4, 3);

julia> u = ScalarField(data, (1.0, 1.0), 1.064; tilts=(0.01, 0.005));

julia> v = ScalarField(data, (1.0, 1.0), 1.064; tilts=([0.01, 0.02, 0.03], 0));
```

See also: [`set_field_data`](@ref), [`power`](@ref), [`normalize_power!`](@ref)
"""
struct ScalarField{U, Nd, S, L, A}
    electric::U
    ds::S
    lambdas::L
    tilts::A

    function ScalarField(u::U, ds::S, lambdas::L,
                         tilts::A) where {U, Nd,
                                          S <: NTuple{Nd},
                                          L <: NamedTuple,
                                          A <: NamedTuple}
        new{U, Nd, S, L, A}(u, ds, lambdas, tilts)
    end

    function ScalarField(u::U, ds::S, lambdas::Union{Real, AbstractArray{<:Real}};
                         tilts::NTuple{Nd, Union{<:Real, <:AbstractArray}}
                         = ntuple(_ -> 0, Nd)) where {Nd, N, S <: NTuple{Nd, Real}, T,
                                                      U <: AbstractArray{Complex{T}, N}}
        @assert Nd in (1, 2)
        @assert N >= Nd
        lambdas = parse_lambdas(u, lambdas, Nd)
        tilts = parse_tilts(u, tilts, Nd)
        L = typeof(lambdas)
        A = typeof(tilts)
        new{U, Nd, S, L, A}(u, ds, lambdas, tilts)
    end

    function ScalarField(nd::NTuple{N, Integer}, ds::NTuple{Nd, Real}, lambdas;
                         tilts = ntuple(_ -> 0, Nd)) where {N, Nd}
        u = zeros(ComplexF64, nd)
        ScalarField(u, ds, lambdas; tilts)
    end
end

Functors.@functor ScalarField (electric,)

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
           for
           (collection, val) in zip(u.tilts.collection, u.tilts.val)])
end

function set_field_ds(u::ScalarField{U, Nd}, ds::NTuple{Nd, Real}) where {U, Nd}
    ScalarField(u.electric, ds, u.lambdas, u.tilts)
end

"""
    set_field_data(u::ScalarField, data::AbstractArray)

Create a new ScalarField with updated field data, preserving all other parameters.

This function creates a copy with new amplitude data while keeping the same spatial grid,
wavelengths, and tilt information.

# Arguments
- `u::ScalarField`: Original field.
- `data::AbstractArray`: New complex field data.

# Returns
New `ScalarField` with updated data.

# Examples
```jldoctest
# Small example for documentation - use larger arrays in practice
julia> u = ScalarField(zeros(ComplexF64, 4, 4), (1.0, 1.0), 1.064);

julia> new_data = rand(ComplexF64, 4, 4);

julia> u_new = set_field_data(u, new_data);
```
"""
function set_field_data(u::ScalarField{U, Nd}, data::V) where {U, V, Nd}
    ScalarField(data, u.ds, u.lambdas.collection; tilts = u.tilts.collection)
end

function set_field_data(u::ScalarField{U, Nd}, data::U) where {U, Nd}
    ScalarField(data, u.ds, u.lambdas, u.tilts)
end

function set_field_tilts(u::ScalarField{U, Nd}, tilts) where {U, Nd}
    ScalarField(u.electric, u.ds, u.lambdas.collection; tilts)
end

# function Base.broadcastable(sf::ScalarField)
#     return Ref(sf)
# end

function Base.broadcasted(f, u::ScalarField)
    ScalarField(complex(broadcast(f, u.electric)),
                u.ds,
                u.lambdas.collection;
                tilts = u.tilts.collection)
end

# function Base.broadcasted(f, a::ScalarField, b::AbstractArray)
#     ScalarField(broadcast(f, a.electric, b), a.lambdas)
# end

function +(u::ScalarField, v::ScalarField)
    set_field_data(u, u.electric + v.electric)
end

Base.getindex(u::ScalarField, i...) = view(u.electric, i...)
Base.size(u::ScalarField) = size(u.electric)
Base.size(u::ScalarField, k::Integer) = size(u.electric, k)

function Base.ndims(u::ScalarField{U, Nd}, spatial::Bool = false) where {U, Nd}
    spatial ? Nd : ndims(u.electric)
end

Base.eltype(u::ScalarField) = eltype(u.electric)

function Base.fill!(u::ScalarField, v)
    u.electric .= v
    u
end

function Base.fill!(u::ScalarField, v::AbstractArray)
    copyto!(u.electric, v)
    u
end

"""
    copy(u::ScalarField)

Create a copy of the scalar field.

Creates a copy of the field data, while sharing the internal representation of other parameters.

# Examples
```jldoctest
# Small example for documentation - use larger arrays in practice
julia> u = ScalarField(zeros(ComplexF64, 4, 4), (1.0, 1.0), 1.064);

julia> u_copy = copy(u);
# Modifying u_copy.electric will not affect u, but the internal arrays representing 
# the wavelengths or tilts must never be modified.
```

See also: [`similar`](@ref)
"""
function Base.copy(u::ScalarField)
    set_field_data(u, copy(u.electric))
end

function Base.copyto!(u::ScalarField, v::ScalarField)
    copyto!(u.electric, v.electric)
    u
end

"""
    similar(u::ScalarField)

Create a new ScalarField with same parameters but uninitialized data.

Useful for creating temporary fields with the same grid, wavelength and tilt
structures as an existing field.

# Examples
```jldoctest
# Small example for documentation - use larger arrays in practice
julia> u = ScalarField(zeros(ComplexF64, 4, 4), (1.0, 1.0), 1.064);

julia> u_tmp = similar(u);  # Same grid/wavelengths/tilts, but data is uninitialized
```

See also: [`copy`](@ref), [`set_field_data`](@ref)
"""
function Base.similar(u::ScalarField)
    set_field_data(u, similar(u.electric))
end

function Base.collect(u::ScalarField)
    collect(u.electric)
end

"""
    vec(u::AbstractArray, nd::Integer=2)
    vec(u::ScalarField)

Convert multi-dimensional arrays or fields into vector of slices.

For AbstractArrays, splits along dimensions beyond the first `nd` spatial dimensions.
For ScalarFields, converts into vector of individual ScalarField objects, each representing
a single slice along non-spatial dimensions. Useful for iteration and visualization.

# Arguments
- `u`: Array or ScalarField to vectorize.
- `nd::Integer`: Number of spatial dimensions (for AbstractArray case only).

# Returns
- For AbstractArrays: Vector of array slices along non-spatial dimensions.
- For ScalarFields: Vector of ScalarField objects, one per slice along non-spatial dimensions.

# Examples

**AbstractArray case:**
```jldoctest
# Small example for documentation - use larger arrays in practice
julia> data = ones(4, 4, 3);  # 2 spatial dims + 1 extra

julia> slices = vec(data, 2);  # Split after 2 spatial dimensions

julia> length(slices)
3

julia> size(slices[1])
(4, 4)
```

**ScalarField case:**
```jldoctest
julia> wavelengths = [0.8, 1.064, 1.550];

# Small example for documentation - use larger arrays in practice
julia> data = zeros(ComplexF64, 4, 4, 3);

julia> u = ScalarField(data, (1.0, 1.0), wavelengths);

julia> u_vec = vec(u);  # Returns Vector{ScalarField} of length 3

julia> length(u_vec)
3

julia> (u_vec[1].lambdas.val, u_vec[2].lambdas.val, u_vec[3].lambdas.val)
(0.8, 1.064, 1.55)
```

See also: [`ScalarField`](@ref), [`eachslice`](@ref)
"""
function Base.vec(u::AbstractArray, nd::Integer)
    @assert nd in (1, 2)
    @assert ndims(u) >= nd
    reshape(eachslice(u; dims = Tuple((nd + 1):ndims(u))), :)
end

function Base.vec(u::ScalarField{U, Nd}) where {U, Nd}
    u_slices = eachslice(u.electric; dims = Tuple((Nd + 1):ndims(u)))
    [ScalarField(data, u.ds, lambda; tilts)
     for (data, lambda, tilts...) in
         bzip(u_slices, u.lambdas.collection, u.tilts.collection...)]
end

"""
    intensity(u::AbstractArray, nd::Integer=2)
    intensity(u::ScalarField)

Compute the total intensity |u|² of optical fields or arrays.

Calculates intensity by summing |u|² over all extra dimensions beyond the spatial ones,
returning the combined intensity distribution for the spatial dimensions only.

# Arguments
- `u::AbstractArray`: Array of complex values.
- `nd::Integer=2`: Number of spatial dimensions (for AbstractArray case only).
- `u::ScalarField`: Optical field with potentially multiple modes.

# Mathematical definition
I[i,j] = Σₖ |u[i,j,k]|² where k runs over all extra dimensions

# Returns
- **AbstractArray**: Array with spatial dimensions only, extra dims summed.
- **ScalarField**: Array with spatial dimensions only, containing total intensity.

# Examples

**AbstractArray case:**
```jldoctest
# Small example for documentation - use larger arrays in practice
julia> data = ones(ComplexF64, 4, 4, 3);  # 2 spatial + 1 extra dim

julia> I = intensity(data, 2);  # Sum over 3rd dimension

julia> size(I)
(4, 4)

julia> I[1,1]
3.0
```

**ScalarField case:**
```jldoctest
# Small example for documentation - use larger arrays in practice
julia> data = ones(ComplexF64, 4, 4, 3);  # 3 modes

julia> u = ScalarField(data, (1.0, 1.0), 1.064);

julia> I = intensity(u);  # Returns 4×4 array (total intensity of all 3 modes)

julia> size(I)
(4, 4)

julia> I[1,1]
3.0
```

See also: [`power`](@ref), [`phase`](@ref), [`abs2`](@ref)
"""
function intensity(u::AbstractArray, nd::Integer = 2)
    @assert nd in (1, 2)
    @assert ndims(u) >= nd
    reshape(sum(abs2, u, dims = Tuple((nd + 1):ndims(u))), size(u)[1:nd])
end

function intensity(u::ScalarField{U, Nd}) where {U, Nd}
    intensity(u.electric)
end

"""
    phase(u::AbstractArray)
    phase(u::ScalarField)

Compute the phase angle of complex arrays or optical fields.

Returns the argument (angle) of complex values in radians, preserving array structure.
For optical fields, this gives the wavefront phase information across all modes.

# Arguments
- `u::AbstractArray`: Array of complex values.
- `u::ScalarField`: Optical field (uses underlying data array).

# Mathematical definition
φ = arg(u) = atan(imag(u), real(u))

# Returns
- **AbstractArray**: Array of same dimensions containing phase values in radians [-π, π].
- **ScalarField**: Calls `phase(u.electric)`, returns array with phase values.

# Examples

**AbstractArray case:**
```jldoctest
julia> data = [1.0+0.0im, 0.0+1.0im, -1.0+0.0im];

julia> phase(data)
3-element Vector{Float64}:
 0.0
 1.5707963267948966
 3.141592653589793
```

**ScalarField case:**
```jldoctest
# Small example for documentation - use larger arrays in practice
julia> data = [1.0+0.0im 0.0+1.0im; -1.0+0.0im 0.0-1.0im];

julia> u = ScalarField(data, (1.0, 1.0), 1.064);

julia> phase(u)
2×2 Matrix{Float64}:
 0.0        1.5708
 3.14159   -1.5708
```

See also: [`intensity`](@ref), [`angle`](@ref)
"""
function phase(u::AbstractArray)
    angle.(u)
end

function phase(u::ScalarField)
    phase(u.electric)
end

"""
    coupling_efficiency(u, v)
    coupling_efficiency(u::ScalarField, v::ScalarField)

Compute the power coupling efficiency between two optical fields.

For multi-dimensional fields, computes power coupling efficiency between corresponding field
distributions (same extra dimension indices). This normalized metric returns values between 0 and 1,
representing the fraction of power that would be transferred from field u to field v.

# Mathematical definition
η = |⟨u,v⟩|² / (‖u‖ ‖v‖)

# Arguments
- `u`: First field (ScalarField or AbstractArray).
- `v`: Second field with same spatial dimensions as `u`.

# Returns
- For AbstractArrays: Scalar coupling efficiency [0,1].
- For ScalarFields: Array of coupling efficiencies, one for each field distribution.

# Examples

**Array case:**
```jldoctest
# Small example for documentation - use larger arrays in practice
julia> u = ones(ComplexF64, 4, 4);

julia> v = ones(ComplexF64, 4, 4);

julia> coupling_efficiency(u, v)
1.0
```

**ScalarField single-mode case:**
```jldoctest
# Small example for documentation - use larger arrays in practice  
julia> field1_data = ones(ComplexF64, 4, 4);

julia> field2_data = ones(ComplexF64, 4, 4);

julia> u = ScalarField(field1_data, (1.0, 1.0), 1.064);

julia> v = ScalarField(field2_data, (1.0, 1.0), 1.064);

julia> coupling_efficiency(u, v)
0-dimensional Array{Float64, 0}:
1.0
```

**ScalarField multi-mode case:**
```jldoctest
# Small example for documentation - use larger arrays in practice
julia> field1_data = ones(ComplexF64, 4, 4, 3);

julia> field2_data = ones(ComplexF64, 4, 4, 3);

julia> u = ScalarField(field1_data, (1.0, 1.0), 1.064);

julia> v = ScalarField(field2_data, (1.0, 1.0), 1.064);

julia> coupling_efficiency(u, v)
3-element Vector{Float64}:
 1.0
 1.0
 1.0
```

See also: [`dot`](@ref), [`power`](@ref), [`PowerCoupling`](@ref)
"""
function coupling_efficiency(u, v)
    abs2(dot(u, v)/(norm(u)*norm(v)))
end

function coupling_efficiency(u::ScalarField{U, Nd}, v::ScalarField{V, Nd}) where {U, V, Nd}
    u_vec = vec(u)
    v_vec = vec(v)
    [coupling_efficiency(u.electric, v.electric) for (u, v) in zip(u_vec, v_vec)]
end

"""
    dot(u::ScalarField, v::ScalarField)

Compute the inner product ⟨u,v⟩ between two optical fields.

For multi-dimensional fields, computes the dot product between corresponding
field distributions (same extra dimension indices).

# Mathematical definition  
⟨u,v⟩ = ∫∫ u*(x,y) v(x,y) dx dy ≈ Σᵢⱼ u*[i,j] v[i,j]

# Arguments
- `u::ScalarField`: First field.
- `v::ScalarField`: Second field with same spatial dimensions as `u`.

# Returns
Vector of complex inner products, one for each field distribution.

# Examples

**Single-mode case:**
```jldoctest
# Small example for documentation - use larger arrays in practice
julia> field1_data = rand(ComplexF64, 4, 4);

julia> field2_data = rand(ComplexF64, 4, 4);

julia> u = ScalarField(field1_data, (1.0, 1.0), 1.064);

julia> v = ScalarField(field2_data, (1.0, 1.0), 1.064);

julia> overlap = dot(u, v);  # 0-dimensional Array storing the complex overlap integral
```

**Multi-mode case:**
```jldoctest
# Small example for documentation - use larger arrays in practice
julia> field1_data = rand(ComplexF64, 4, 4, 3);

julia> field2_data = rand(ComplexF64, 4, 4, 3);

julia> u = ScalarField(field1_data, (1.0, 1.0), 1.064);

julia> v = ScalarField(field2_data, (1.0, 1.0), 1.064);

julia> overlaps = dot(u, v);  # 3-element Vector of complex overlaps
```

See also: [`correlation`](@ref), [`power`](@ref)
"""
function LinearAlgebra.dot(u::ScalarField{U, Nd}, v::ScalarField{V, Nd}) where {U, V, Nd}
    u_vec = vec(u)
    v_vec = vec(v)
    [dot(u.electric, v.electric) for (u, v) in zip(u_vec, v_vec)]
end

"""
    power(u::ScalarField)

Compute the optical power of the field.

Calculates the spatial integral of the intensity |u|² over the field domain,
properly accounting for the spatial sampling.

# Mathematical definition
P = ∫∫ |u(x,y)|² dx dy ≈ Σᵢⱼ |u[i,j]|² × dx × dy.

# Returns  
Array of power value(s) with same dimensions as `u`, spatial dimensions being reduced to size 1.

# Examples
```jldoctest
# Small example for documentation - use larger arrays in practice
julia> data = rand(ComplexF64, 4, 4, 3);

julia> u = ScalarField(data, (1.0, 1.0), 1.064);

julia> P = power(u);  # Returns 1×1×3 Array
```

See also: [`normalize_power!`](@ref)
"""
function power(u::ScalarField{U, Nd}) where {U, Nd}
    dims = ntuple(k -> k, Nd)
    sum(abs2, u.electric; dims = dims) .* prod(u.ds)
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
```jldoctest
# Small example for documentation - use larger arrays in practice
julia> u = ScalarField(rand(ComplexF64, 4, 4, 3), (1.0, 1.0), 1.064);

julia> normalize_power!(u);        # Normalize all fields to 1 W

julia> normalize_power!(u, 1e-3);  # Normalize all fields to 1 mW

julia> power(u)
1×1×3 Array{Float64, 3}:
[:, :, 1] =
 0.001

[:, :, 2] =
 0.001

[:, :, 3] =
 0.001

# For multiple fields: normalize each field separately
julia> normalize_power!(u, reshape([1e-3, 1.8e-3, 3e-3], 1, 1, 3));  # Different power per field

julia> power(u)
1×1×3 Array{Float64, 3}:
[:, :, 1] =
 0.001

[:, :, 2] =
 0.0018

[:, :, 3] =
 0.003
```

See also: [`power`](@ref)
"""
function normalize_power!(u::ScalarField, target_power = 1)
    u.electric .*= sqrt.(target_power ./ power(u))
    u
end

function Base.conj(u::ScalarField)
    set_field_data(u, conj(u.electric))
end

end
