module OpticalComponents

using Functors
using ..GridUtils
using ..Types

export propagate!, propagate
export propagate_and_save!, propagate_and_save
export backpropagate!, backpropagate
export backpropagate_with_gradient!, backpropagate_with_gradient

abstract type AbstractPropagator{M <: Trainability} <: AbstractOpticalComponent{M} end

trainable(p::AbstractOpticalComponent{Static}) = NamedTuple{}()

function trainable(p::AbstractOpticalComponent{<:Trainable})
    error("Not implemented")
end

function get_preallocated_gradient(p::AbstractOpticalComponent{<:Trainable{<:NamedTuple}})
    error("Not implemented")
end

function propagate!(u, p::AbstractOpticalComponent, direction::Type{<:Direction})
    error("Not implemented")
end

function propagate(p::AbstractOpticalComponent, direction::Type{<:Direction})
    error("Not implemented")
end

function propagate_and_save!(u, p::AbstractOpticalComponent{<:Trainable},
        direction::Type{<:Direction})
    error("Not implemented")
end

function propagate_and_save(p::AbstractOpticalComponent{<:Trainable},
        direction::Type{<:Direction})
    error("Not implemented")
end

function backpropagate!(u, p::AbstractOpticalComponent, direction::Type{<:Direction})
    error("Not implemented")
end

function backpropagate_with_gradient!(
        ∂v, ∂p::NamedTuple, p::AbstractOpticalComponent{<:Trainable},
        direction::Type{<:Direction})
    error("Not implemented")
end

function propagate(u, p::AbstractOpticalComponent, direction::Type{<:Direction})
    propagate!(copy(u), p, direction)
end

function propagate_and_save(u, p::AbstractOpticalComponent{<:Trainable},
        direction::Type{<:Direction})
    propagate_and_save!(copy(u), p, direction)
end

function backpropagate(u, p::AbstractOpticalComponent, direction::Type{<:Direction})
    backpropagate!(copy(u), p, direction)
end

function backpropagate_with_gradient!(
        ∂v, p::AbstractOpticalComponent{Trainable{Nothing}},
        direction::Type{<:Direction})
    ∂p = fmap(similar, trainable(p))
    backpropagate_with_gradient!(∂v, ∂p, p, direction)
end

function backpropagate_with_gradient!(
        ∂v, p::AbstractOpticalComponent{<:Trainable{<:NamedTuple}},
        direction::Type{<:Direction})
    ∂p = get_preallocated_gradient(p)
    backpropagate_with_gradient!(∂v, ∂p, p, direction)
end

function backpropagate_with_gradient(
        ∂v, p::AbstractOpticalComponent{<:Trainable},
        direction::Type{<:Direction})
    backpropagate_with_gradient!(copy(∂v), p, direction)
end

include("freespace.jl")
export ASProp, RSProp

include("phasemask.jl")
export Phase

include("seeder.jl")
export Seeder

end
