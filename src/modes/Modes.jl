module Modes

using ..GridUtils

abstract type Mode{N, T} <: Function end

function Base.eltype(m::Mode{N, T}) where {N, T <: Real}
    Complex{T}
end

function eval_mode(m::Mode, x...)
    error("Not implemented for $(typeof(m))")
end

function (m::Mode{N})(x::Vararg{Real, N}) where {N}
    eval_mode(m, x...)
end

function (m::Mode{1})(x_vec::AbstractVector; xc = 0.0)
    if !iszero(xc)
        x_vec = x_vec .+ xc
    end
    [eval_mode(m, x) for x in x_vec]
end

function (m::Mode{2, T})(u::AbstractArray{Complex{T}, 2},
        x_vec::AbstractVector,
        y_vec::AbstractVector,
        t::AbstractAffineMap = Id2D()) where {T}
    @assert size(u, 1) == length(x_vec)
    @assert size(u, 2) == length(y_vec)
    t = inv(t)
    @inbounds for (j, y) in enumerate(y_vec)
        for (i, x) in enumerate(x_vec)
            u[i, j] = eval_mode(m, t(x, y)...)
        end
    end
    u
end

function (m::Mode{2, T})(x_vec::AbstractVector, y_vec::AbstractVector,
        t::AbstractAffineMap = Id2D()) where {T}
    nx = length(x_vec)
    ny = length(y_vec)
    u = zeros(Complex{T}, (nx, ny))
    m(u, x_vec, y_vec, t)
end

abstract type LKind end

struct Vortex <: LKind end
struct Even <: LKind end
struct Odd <: LKind end

function parse_kind(kind::Symbol)
    if kind == :vortex
        return Vortex()
    elseif kind == :even
        return Even()
    elseif kind == :odd
        return Odd()
    else
        error("Unknown kind: $kind")
    end
end

include("gaussian_modes.jl")
export Gaussian1D, Gaussian, HermiteGaussian1D, HermiteGaussian, LaguerreGaussian
export hermite_gaussian_groups

using AbstractFFTs
include("speckle_generator.jl")
export generate_speckle

include("layouts.jl")
export PointLayout, GridLayout, TriangleLayout, CustomLayout
export generate_mode_stack

end
