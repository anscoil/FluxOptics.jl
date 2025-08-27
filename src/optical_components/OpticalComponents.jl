module OpticalComponents

using Functors
using AbstractFFTs
using ..GridUtils
using ..Fields
using ..FFTutils

export Direction, Forward, Backward
export GradientAllocation, GradNoAlloc, GradAllocated
export Trainability, Trainable, Static
export AbstractOpticalComponent, AbstractOpticalSource
export AbstractCustomComponent, AbstractCustomSource
export AbstractPureComponent, AbstractPureSource
export propagate!, propagate
export propagate_and_save!, propagate_and_save
export backpropagate!, backpropagate
export backpropagate_with_gradient!, backpropagate_with_gradient

abstract type Direction end
struct Forward <: Direction end
struct Backward <: Direction end

function Base.reverse(::Type{Forward})
    Backward
end

function Base.reverse(::Type{Backward})
    Forward
end

abstract type GradientAllocation end
struct GradNoAlloc <: GradientAllocation end
struct GradAllocated <: GradientAllocation end

abstract type Trainability end
struct Static <: Trainability end
struct Trainable{A <: GradientAllocation} <: Trainability end

abstract type AbstractOpticalComponent{M <: Trainability} end

trainable(p::AbstractOpticalComponent{Static}) = NamedTuple{}()

function trainable(p::AbstractOpticalComponent{<:Trainable})
    error("Not implemented")
end

abstract type AbstractCustomComponent{M} <: AbstractOpticalComponent{M} end

function get_preallocated_gradient(p::AbstractCustomComponent{<:Trainable{GradNoAlloc}})
    fmap(similar, trainable(p))
end

function get_preallocated_gradient(p::AbstractCustomComponent{<:Trainable{GradAllocated}})
    error("Not implemented")
end

function propagate!(u, p::AbstractCustomComponent, direction::Type{<:Direction})
    error("Not implemented")
end

function propagate_and_save!(u, p::AbstractCustomComponent{<:Trainable},
        direction::Type{<:Direction})
    error("Not implemented")
end

function backpropagate!(u, p::AbstractCustomComponent, direction::Type{<:Direction})
    error("Not implemented")
end

function backpropagate_with_gradient!(
        ∂v, ∂p::NamedTuple, p::AbstractCustomComponent{<:Trainable},
        direction::Type{<:Direction})
    error("Not implemented")
end

function propagate(u, p::AbstractCustomComponent, direction::Type{<:Direction})
    propagate!(copy(u), p, direction)
end

function propagate(u::AbstractArray, p::AbstractCustomComponent,
        λ::Real, direction::Type{<:Direction})
    propagate!(copy(u), p, λ, direction)
end

function propagate_and_save(u, p::AbstractCustomComponent{<:Trainable},
        direction::Type{<:Direction})
    propagate_and_save!(copy(u), p, direction)
end

function backpropagate(u, p::AbstractCustomComponent, direction::Type{<:Direction})
    backpropagate!(copy(u), p, direction)
end

function backpropagate_with_gradient!(
        ∂v, p::AbstractCustomComponent{<:Trainable},
        direction::Type{<:Direction})
    ∂p = get_preallocated_gradient(p)
    backpropagate_with_gradient!(∂v, ∂p, p, direction)
end

function backpropagate_with_gradient(
        ∂v, p::AbstractCustomComponent{<:Trainable},
        direction::Type{<:Direction})
    backpropagate_with_gradient!(copy(∂v), p, direction)
end

abstract type AbstractPureComponent{M} <: AbstractOpticalComponent{M} end

function propagate(u, p::AbstractPureComponent, direction::Type{<:Direction})
    error("Not implemented")
end

function propagate!(u, p::AbstractPureComponent, direction::Type{<:Direction})
    propagate(u, p, direction)
end

abstract type AbstractOpticalSource{M} <: AbstractOpticalComponent{M} end

function propagate(p::AbstractOpticalSource, direction::Type{<:Direction})
    error("Not implemented")
end

abstract type AbstractPureSource{M} <: AbstractOpticalSource{M} end

abstract type AbstractCustomSource{M} <: AbstractOpticalSource{M} end

function get_preallocated_gradient(p::AbstractCustomSource{<:Trainable{GradNoAlloc}})
    fmap(similar, trainable(p))
end

function get_preallocated_gradient(p::AbstractCustomSource{<:Trainable{GradAllocated}})
    error("Not implemented")
end

function propagate_and_save(p::AbstractCustomSource{<:Trainable},
        direction::Type{<:Direction})
    error("Not implemented")
end

function backpropagate_with_gradient!(
        ∂v, ∂p::NamedTuple, p::AbstractCustomSource{<:Trainable},
        direction::Type{<:Direction})
    error("Not implemented")
end

function backpropagate_with_gradient(
        ∂v, p::AbstractCustomSource{<:Trainable},
        direction::Type{<:Direction})
    ∂p = get_preallocated_gradient(p)
    backpropagate_with_gradient!(∂v, ∂p, p, direction)
end

include("abstract_kernel.jl")

include("freespace.jl")
export ASProp, ASPropZ, RSProp, CollinsProp, FourierLens, ParaxialProp

include("scalar_source.jl")
export ScalarSource

include("phasemask.jl")
export Phase

include("tea_doe.jl")
export TeaDOE, TeaReflector

end
