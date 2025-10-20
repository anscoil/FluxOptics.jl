module Fields

using Functors
using LinearAlgebra
using StaticArrays
using ..FluxOptics
using ..FluxOptics: isbroadcastable, bzip

import Base: +, -, *, /

export ScalarField
export get_lambdas, get_lambdas_collection
export get_tilts, get_tilts_collection, offset_tilts!
export select_lambdas, select_tilts, set_field_ds!, set_field_data, set_field_tilts
export is_on_axis
export power, normalize_power!, coupling_efficiency, intensity, phase

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
julia> # Small example for documentation - use larger arrays in practice

julia> data = rand(ComplexF64, 4, 4);

julia> u = ScalarField(data, (1.0, 1.0), 1.064);

julia> size(u)
(4, 4)
```

**Creating zero-initialized field:**
```jldoctest
julia> # Small example for documentation - use larger arrays in practice

julia> u = ScalarField((4, 4), (1.0, 1.0), 1.064);

julia> size(u)
(4, 4)
```

**Multi-wavelength field:**
```jldoctest
julia> wavelengths = [0.8, 1.064, 1.550];

julia> # Small example for documentation - use larger arrays in practice

julia> data = zeros(ComplexF64, 4, 4, 3);

julia> u = ScalarField(data, (1.0, 1.0), wavelengths);
```

**Field with initial tilt:**
```jldoctest
julia> wavelengths = [0.8, 1.064, 1.55];

julia> # Small example for documentation - use larger arrays in practice

julia> data = zeros(ComplexF64, 4, 4, 3);

julia> u = ScalarField(data, (1.0, 1.0), 1.064; tilts=(0.01, 0.005));

julia> v = ScalarField(data, (1.0, 1.0), 1.064; tilts=([0.01, 0.02, 0.03], 0));
```

See also: [`set_field_data`](@ref), [`power`](@ref), [`normalize_power!`](@ref)
"""
struct ScalarField{U, Nd, L, A}
    electric::U
    ds::MVector{Nd, Float64}
    lambdas::L
    tilts::A

    function ScalarField(u::U, ds::S, lambdas::L,
                         tilts::A) where {U, Nd, S <: MVector{Nd}, L <: NamedTuple,
                                          A <: NamedTuple}
        new{U, Nd, L, A}(u, ds, lambdas, tilts)
    end

    function ScalarField(u::U, ds::NTuple{Nd, Real},
                         lambdas::Union{Real, AbstractArray{<:Real}};
                         tilts::NTuple{Nd, Union{<:Real, <:AbstractArray}}
                         = ntuple(_ -> 0, Nd)) where {Nd, N, T,
                                                      U <: AbstractArray{Complex{T}, N}}
        @assert Nd in (1, 2)
        @assert N >= Nd
        lambdas = parse_lambdas(u, lambdas, Nd)
        tilts = parse_tilts(u, tilts, Nd)
        L = typeof(lambdas)
        A = typeof(tilts)
        new{U, Nd, L, A}(u, ds, lambdas, tilts)
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

function set_field_ds!(u::ScalarField{U, Nd}, ds::NTuple{Nd, Real}) where {U, Nd}
    u.ds .= ds
    u
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
julia> # Small example for documentation - use larger arrays in practice

julia> u = ScalarField(zeros(ComplexF64, 4, 4), (1.0, 1.0), 1.064);

julia> new_data = rand(ComplexF64, 4, 4);

julia> u_new = set_field_data(u, new_data);
```
"""
function set_field_data(u::ScalarField{U, Nd}, data::V) where {U, V, Nd}
    ScalarField(data, Tuple(u.ds), u.lambdas.collection; tilts = u.tilts.collection)
end

# function set_field_data(u::ScalarField{U, Nd}, data::U) where {U, Nd}
#     ScalarField(data, copy(u.ds), u.lambdas, u.tilts)
# end

"""
    set_field_tilts(u::ScalarField, tilts) -> ScalarField

Create a new ScalarField with updated tilts, preserving all other parameters.

This function creates a copy with new tilt values while keeping the same field data,
spatial grid, and wavelengths.

# Arguments
- `u::ScalarField`: Original field.
- `tilts`: New tilt values as `(θx, θy)` tuple, where each component can be scalar or array.

# Returns
New `ScalarField` with updated tilts.

# Examples
```jldoctest
julia> u = ScalarField(ones(ComplexF64, 4, 4), (1.0, 1.0), 1.064; tilts=(0.01, 0.0));

julia> u_tilted = set_field_tilts(u, (0.02, 0.01));
```
"""
function set_field_tilts(u::ScalarField{U, Nd}, tilts) where {U, Nd}
    ScalarField(u.electric, Tuple(u.ds), u.lambdas.collection; tilts)
end

"""
    is_on_axis(u::ScalarField) -> Bool

Check if the field has zero tilts (on-axis propagation).

Returns `true` if all tilt components are zero, `false` otherwise.
"""
function is_on_axis(u::ScalarField)
    all(iszero, u.tilts.collection)
end

"""
    offset_tilts!(u::ScalarField, tilts)

Add offset to existing tilts and apply corresponding linear phase to the field in-place.

This function modifies both the tilt metadata and the electric field by:
1. Adding the offset to the stored tilt values
2. Multiplying the field by `exp(i 2π/λ (Δθx⋅x + Δθy⋅y))` to maintain consistency

This shifts the reference frame without rotating the angular spectrum.

# Arguments
- `u::ScalarField`: Field to modify.
- `tilts`: Tilt offsets as `(Δθx, Δθy)` tuple in radians.

# Examples
```jldoctest
julia> u = ScalarField(ones(ComplexF64, 4, 4), (1.0, 1.0), 1.064; tilts=(0.01, 0.0));

julia> offset_tilts!(u, (0.005, 0.005));
```
"""
function offset_tilts!(u::ScalarField{U, Nd},
                       tilts::NTuple{Nd, Union{<:Real, <:AbstractArray}}) where {U, Nd}
    if tilts == u.tilts.collection
        return u
    end
    tilts = parse_tilts(u.electric, tilts, Nd)
    θx0, θy0 = u.tilts.val
    θx, θy = tilts.val
    xv, yv = spatial_vectors(size(u)[1:Nd]..., u.ds...)
    @. u.electric *= cis(2π/u.lambdas.val*((sin(θx0)-sin(θx))*xv + (sin(θy0)-sin(θy))*yv'))
    foreach(((y, x),) -> copyto!(y, x), zip(u.tilts.val, tilts.val))
    foreach(((y, x),) -> copyto!(y, x), zip(u.tilts.collection, tilts.collection))
    u
end

# function Base.broadcastable(sf::ScalarField)
#     return Ref(sf)
# end

"""
    broadcasted(f::Function, u::ScalarField) -> ScalarField

Apply function `f` element-wise to the electric field. Returns a new ScalarField with transformed data.
"""
function Base.broadcasted(f::Function, u::ScalarField)
    ScalarField(complex(broadcast(f, u.electric)),
                Tuple(u.ds),
                u.lambdas.collection;
                tilts = u.tilts.collection)
end

# function Base.broadcasted(f, a::ScalarField, b::AbstractArray)
#     ScalarField(broadcast(f, a.electric, b), a.lambdas)
# end

function +(u::ScalarField, v::ScalarField)
    set_field_data(u, u.electric + v.electric)
end

function -(u::ScalarField, v::ScalarField)
    set_field_data(u, u.electric - v.electric)
end

function -(u::ScalarField)
    set_field_data(u, -u.electric)
end

function *(a::Number, u::ScalarField)
    set_field_data(u, a .* u.electric)
end

function *(u::ScalarField, a::Number)
    a * u
end

function /(u::ScalarField, a::Number)
    set_field_data(u, u.electric ./ a)
end

"""
    getindex(u::ScalarField, i...)

Access elements of the electric field. Returns a view into the field data.
"""
Base.getindex(u::ScalarField, i...) = view(u.electric, i...)

"""
    size(u::ScalarField) -> Tuple
    size(u::ScalarField, k::Integer) -> Int

Return the size of the electric field array.
"""
Base.size(u::ScalarField) = size(u.electric)
Base.size(u::ScalarField, k::Integer) = size(u.electric, k)

"""
    ndims(u::ScalarField, spatial::Bool=false) -> Int

Return number of dimensions. If `spatial=true`, returns only spatial dimensions (Nd), otherwise returns total dimensions of the electric field array.
"""
function Base.ndims(u::ScalarField{U, Nd}, spatial::Bool = false) where {U, Nd}
    spatial ? Nd : ndims(u.electric)
end

"""
    eltype(u::ScalarField) -> Type

Return the element type of the electric field.
"""
Base.eltype(u::ScalarField) = eltype(u.electric)

"""
    fill!(u::ScalarField, v) -> ScalarField

Fill the electric field with value `v` in-place.

# Arguments
- `u::ScalarField`: Field to modify.
- `v`: Fill value, either a scalar (fills entire field) or array (copied into field).

# Returns
Modified `ScalarField` (same as input).

# Examples
```jldoctest
julia> u = ScalarField((4, 4), (1.0, 1.0), 1.064);

julia> fill!(u, 1.0 + 0.0im);  # Fill with constant

julia> fill!(u, rand(ComplexF64, 4, 4));  # Fill with array
```
"""
function Base.fill!(u::ScalarField, v::Number)
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
julia> # Small example for documentation - use larger arrays in practice

julia> u = ScalarField(zeros(ComplexF64, 4, 4), (1.0, 1.0), 1.064);

julia> u_copy = copy(u);

julia> # Modifying u_copy.electric will not affect u, but the internal arrays representing

julia> # the wavelengths or tilts must never be modified.
```

See also: [`similar`](@ref similar(::ScalarField))
"""
function Base.copy(u::ScalarField)
    ScalarField(copy(u.electric), copy(u.ds), deepcopy(u.lambdas), deepcopy(u.tilts))
end

"""
    copyto!(u::ScalarField, v::ScalarField) -> ScalarField

Copy data from `v` into `u`. Modifies `u` in-place and returns it.
"""
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
julia> # Small example for documentation - use larger arrays in practice

julia> u = ScalarField(zeros(ComplexF64, 4, 4), (1.0, 1.0), 1.064);

julia> u_tmp = similar(u);  # Same grid/wavelengths/tilts, but data is uninitialized
```

See also: [`copy`](@ref copy(::ScalarField)), [`set_field_data`](@ref)
"""
function Base.similar(u::ScalarField)
    ScalarField(similar(u.electric), copy(u.ds), deepcopy(u.lambdas), deepcopy(u.tilts))
end

"""
    collect(u::ScalarField) -> Array

Convert field data to a regular CPU array. Returns an `Array` (not a `ScalarField`).

Useful for converting GPU arrays (e.g., `CuArray`) to CPU for analysis, plotting, or saving.

# Examples
```jldoctest
julia> u = ScalarField(ones(ComplexF64, 4, 4), (1.0, 1.0), 1.064);

julia> data = collect(u);  # Returns Array, not ScalarField

julia> typeof(data)
Matrix{ComplexF64} (alias for Array{Complex{Float64}, 2})
```
"""
function Base.collect(u::ScalarField)
    collect(u.electric)
end

function Base.vec(u::AbstractArray, nd::Integer)
    @assert nd in (1, 2)
    @assert ndims(u) >= nd
    reshape(eachslice(u; dims = Tuple((nd + 1):ndims(u))), :)
end

"""
    vec(u::AbstractArray, nd::Integer=2)
    vec(u::ScalarField)

Convert multi-dimensional arrays or fields into vector of slices.

For AbstractArray, splits along dimensions beyond the first `nd` spatial dimensions.
For ScalarField, converts into vector of individual ScalarField objects, each representing
a single slice along non-spatial dimensions. Useful for iteration and visualization.

# Arguments
- `u`: Array or ScalarField to vectorize.
- `nd::Integer`: Number of spatial dimensions (for AbstractArray case only).

# Returns
- For AbstractArray: Vector of array slices along non-spatial dimensions.
- For ScalarField: Vector of ScalarField objects, one per slice along non-spatial dimensions.

# Examples

**AbstractArray case:**
```jldoctest
julia> # Small example for documentation - use larger arrays in practice

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

julia> # Small example for documentation - use larger arrays in practice

julia> data = zeros(ComplexF64, 4, 4, 3);

julia> u = ScalarField(data, (1.0, 1.0), wavelengths);

julia> u_vec = vec(u);  # Returns Vector{ScalarField} of length 3

julia> length(u_vec)
3

julia> (u_vec[1].lambdas.val, u_vec[2].lambdas.val, u_vec[3].lambdas.val)
(0.8, 1.064, 1.55)
```

See also: [`ScalarField`](@ref)
"""
function Base.vec(u::ScalarField{U, Nd}) where {U, Nd}
    u_slices = eachslice(u.electric; dims = Tuple((Nd + 1):ndims(u)))
    [ScalarField(data, Tuple(u.ds), lambda; tilts)
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
julia> # Small example for documentation - use larger arrays in practice

julia> data = ones(ComplexF64, 4, 4, 3);  # 2 spatial + 1 extra dim

julia> I = intensity(data, 2);  # Sum over 3rd dimension

julia> size(I)
(4, 4)

julia> I[1,1]
3.0
```

**ScalarField case:**
```jldoctest
julia> # Small example for documentation - use larger arrays in practice

julia> data = ones(ComplexF64, 4, 4, 3);  # 3 modes

julia> u = ScalarField(data, (1.0, 1.0), 1.064);

julia> I = intensity(u);  # Returns 4×4 array (total intensity of all 3 modes)

julia> size(I)
(4, 4)

julia> I[1,1]
3.0
```

See also: [`power`](@ref), [`phase`](@ref)
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
julia> # Small example for documentation - use larger arrays in practice

julia> data = [1.0+0.0im 0.0+1.0im; -1.0+0.0im 0.0-1.0im];

julia> u = ScalarField(data, (1.0, 1.0), 1.064);

julia> phase(u)
2×2 Matrix{Float64}:
 0.0       1.5708
 3.14159  -1.5708
```

See also: [`intensity`](@ref), [`phase`](@ref)
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
julia> # Small example for documentation - use larger arrays in practice

julia> u = ones(ComplexF64, 4, 4);

julia> v = ones(ComplexF64, 4, 4);

julia> coupling_efficiency(u, v)
1.0
```

**ScalarField single-mode case:**
```jldoctest
julia> # Small example for documentation - use larger arrays in practice

julia> field1_data = ones(ComplexF64, 4, 4);

julia> field2_data = ones(ComplexF64, 4, 4);

julia> u = ScalarField(field1_data, (1.0, 1.0), 1.064);

julia> v = ScalarField(field2_data, (1.0, 1.0), 1.064);

julia> coupling_efficiency(u, v)
1-element Vector{Float64}:
 1.0
```

**ScalarField multi-mode case:**
```jldoctest
julia> # Small example for documentation - use larger arrays in practice

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

See also: [`power`](@ref)
"""
function coupling_efficiency(u::AbstractArray, v::AbstractArray)
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
julia> # Small example for documentation - use larger arrays in practice

julia> field1_data = rand(ComplexF64, 4, 4);

julia> field2_data = rand(ComplexF64, 4, 4);

julia> u = ScalarField(field1_data, (1.0, 1.0), 1.064);

julia> v = ScalarField(field2_data, (1.0, 1.0), 1.064);

julia> overlap = dot(u, v);  # 0-dimensional Array storing the complex overlap integral
```

**Multi-mode case:**
```jldoctest
julia> # Small example for documentation - use larger arrays in practice

julia> field1_data = rand(ComplexF64, 4, 4, 3);

julia> field2_data = rand(ComplexF64, 4, 4, 3);

julia> u = ScalarField(field1_data, (1.0, 1.0), 1.064);

julia> v = ScalarField(field2_data, (1.0, 1.0), 1.064);

julia> overlaps = dot(u, v);  # 3-element Vector of complex overlaps
```

See also: [`coupling_efficiency`](@ref), [`dot`](@ref dot(::ScalarField)), [`power`](@ref)
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
julia> # Small example for documentation - use larger arrays in practice

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
julia> # Small example for documentation - use larger arrays in practice

julia> u = ScalarField(rand(ComplexF64, 4, 4, 3), (1.0, 1.0), 1.064);

julia> normalize_power!(u);        # Normalize all fields to 1 W

julia> normalize_power!(u, 1e-3);  # Normalize all fields to 1 mW

julia> all(x -> isapprox(x, 1e-3), power(u))
true

julia> # For multiple fields: normalize each field separately

julia> power_values = reshape([1e-3, 2e-3, 3e-3], 1, 1, 3);

julia> normalize_power!(u, power_values);  # Different power per field

julia> isapprox(power(u), power_values)
true
```

See also: [`power`](@ref)
"""
function normalize_power!(u::ScalarField, target_power = 1)
    u.electric .*= sqrt.(target_power ./ power(u))
    u
end

"""
    conj(u::ScalarField) -> ScalarField

Return complex conjugate of the field. Creates a new field with conjugated electric field values.
"""
function Base.conj(u::ScalarField)
    set_field_data(u, conj(u.electric))
end

end
