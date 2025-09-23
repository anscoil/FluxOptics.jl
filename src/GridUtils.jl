module GridUtils

using StaticArrays
using Rotations
using CoordinateTransformations

export spatial_vectors
export AbstractAffineMap, Shift2D, Rot2D, Id2D

function spatial_vectors(ns::NTuple{Nd, Real}, ds::NTuple{Nd, Real};
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

function Base.inv(t::Id2D)
    t
end

end
