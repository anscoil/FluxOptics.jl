module GridUtils

using StaticArrays
using Rotations
using CoordinateTransformations

export spatial_vectors
export AbstractAffineMap, Shift2D, Rot2D, Id2D

function spatial_vectors(nx, ny, dx, dy; xc = 0.0, yc = 0.0)
    x_vec = ((0:(nx - 1)) .- (nx-1)/2)*dx .+ xc
    y_vec = ((0:(ny - 1)) .- (ny-1)/2)*dy .+ yc
    (x_vec, y_vec)
end

const Shift2D{T} = Translation{SVector{2, T}}

function Shift2D(x::T1, y::T2) where {T1 <: Real, T2 <: Real}
    Translation{SVector{2, promote_type(T1, T2)}}(@SVector [x, y])
end

const Rot2D{T} = LinearMap{RotMatrix2{T}}

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

end
