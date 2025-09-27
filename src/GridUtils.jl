module GridUtils

using StaticArrays
using Rotations
using CoordinateTransformations

export spatial_vectors
export AbstractAffineMap, Shift2D, Rot2D, Id2D

"""
    spatial_vectors(ns::NTuple{Nd,Real}, ds::NTuple{Nd,Real}; offset::NTuple{Nd,Real}=ntuple(_->0, Nd))
    spatial_vectors(nx, dx; xc=0.0)
    spatial_vectors(nx, ny, dx, dy; xc=0.0, yc=0.0)

Generate spatial coordinate vectors for optical field grids.

Creates coordinate arrays centered at zero with specified sampling and optional offset.
Essential for defining spatial grids in optical simulations.

# Arguments
- `ns::NTuple`: Number of points in each dimension.
- `ds::NTuple`: Spatial sampling interval in each dimension (user-defined unit).
- `offset::NTuple=ntuple(_->0, Nd)`: Offset of the zero position from the center of the grid, for each dimension.
  The center of the grid is then -offset.

# Returns
Tuple of coordinate vectors, one per spatial dimension. Each coordinate vector is an AbstractRange.

# Examples
```jldoctest
# 1D case
julia> x, = spatial_vectors(4, 1.0);

julia> isa(x, AbstractRange)
true

julia> collect(x)
4-element Vector{Float64}:
 -1.5
 -0.5
  0.5
  1.5

# 2D case with offset
julia> x, y = spatial_vectors(4, 4, 1.0, 1.0; xc=2.0);

julia> collect(x)
4-element Vector{Float64}:
 -3.5
 -2.5
 -1.5
 -0.5
```

See also: [`Shift2D`](@ref), [`Rot2D`](@ref)
"""
function spatial_vectors(ns::NTuple{Nd, Real},
                         ds::NTuple{Nd, Real};
                         offset::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)) where {Nd}
    Tuple([((0:(nx - 1)) .- (nx-1)/2)*dx .- xc for (nx, dx, xc) in zip(ns, ds, offset)])
end

function spatial_vectors(nx, dx; xc = 0.0)
    spatial_vectors((nx,), (dx,); offset = (xc,))
end

function spatial_vectors(nx, ny, dx, dy; xc = 0.0, yc = 0.0)
    spatial_vectors((nx, ny), (dx, dy); offset = (xc, yc))
end

const Shift2D{T} = Translation{SVector{2, T}}

"""
    Shift2D(x::Real, y::Real)

Create a 2D translation transformation.

Represents a spatial translation by (x, y) in micrometers. Can be composed 
with other transformations using the ∘ operator.

# Arguments
- `x::Real`: Translation in x direction.
- `y::Real`: Translation in y direction.

# Examples
```jldoctest
julia> t = Shift2D(2.0, 3.0);

julia> t([1.0, 1.0])  # Apply transformation
2-element StaticArrays.SVector{2, Float64}:
 3.0
 4.0

# Composition with rotation
julia> combined = Shift2D(1.0, 0.0) ∘ Rot2D(π/4);

julia> combined([1.0, 0.0])  # Rotate then translate
2-element StaticArrays.SVector{2, Float64}:
 1.7071067811865475
 0.7071067811865475
```

See also: [`Rot2D`](@ref), [`Id2D`](@ref), [`∘`](@ref)
"""
function Shift2D(x::T1, y::T2) where {T1 <: Real, T2 <: Real}
    Translation{SVector{2, promote_type(T1, T2)}}(@SVector [x, y])
end

const Rot2D{T} = LinearMap{RotMatrix2{T}}

"""
    Rot2D(θ::Real)

Create a 2D rotation transformation.

Represents a rotation by angle θ (in radians) around the origin. 
Can be composed with other transformations using the ∘ operator.

# Arguments
- `θ::Real`: Rotation angle in radians (positive = counterclockwise).

# Examples
```jldoctest
julia> r = Rot2D(π/2);  # 90° rotation

julia> r([1.0, 0.0])  # Rotate point
2-element StaticArrays.SVector{2, Float64}:
  6.123233995736766e-17
  1.0

# Composition: translate then rotate
julia> transform = Rot2D(π/4) ∘ Shift2D(2.0, 0.0);

julia> transform([0.0, 0.0])
2-element StaticArrays.SVector{2, Float64}:
 1.4142135623730951
 1.414213562373095
```

See also: [`Shift2D`](@ref), [`Id2D`](@ref), [`∘`](@ref)
"""
function Rot2D(θ::T) where {T <: Real}
    LinearMap(RotMatrix2{T}(θ))
end

struct Id2D <: AbstractAffineMap end

function CoordinateTransformations.compose(id::Id2D, t::AbstractAffineMap)
    t
end

function CoordinateTransformations.compose(t::AbstractAffineMap, id::Id2D)
    t
end

function (t::AbstractAffineMap)(x, y)
    t(@SVector [x, y])
end

function (t::Id2D)(x, y)
    @SVector [x, y]
end

function Base.inv(t::Id2D)
    t
end

end
