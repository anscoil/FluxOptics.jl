module OpticalComponents

using Functors
using ..GridUtils
using ..Fields

export Direction, Forward, Backward
export Trainability, Trainable, Static
export is_trainable, has_prealloc
export AbstractOpticalComponent, AbstractPropagator, AbstractOpticalSource
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

abstract type Trainability end
struct Static <: Trainability end
struct Trainable{A <: Union{Nothing, <:NamedTuple}} <: Trainability end

is_trainable(::Type{<:Trainable}) = true
is_trainable(::Type{Static}) = false

has_prealloc(::Type{Trainable{Nothing}}) = false
has_prealloc(::Type{Trainable{<:NamedTuple}}) = true

abstract type AbstractOpticalComponent{M <: Trainability} end
abstract type AbstractPropagator{M <: Trainability} <: AbstractOpticalComponent{M} end
abstract type AbstractOpticalSource{M <: Trainability} <: AbstractOpticalComponent{M} end

function adapt_dim(A::Type{<:AbstractArray{T}}, n::Integer, f = identity) where {T}
    @assert isconcretetype(A)
    A.name.wrapper{f(A.parameters[1]), n, A.parameters[3:end]...}
end

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

function propagate(p::AbstractOpticalSource, direction::Type{<:Direction})
    error("Not implemented")
end

function propagate_and_save!(u, p::AbstractOpticalComponent{<:Trainable},
        direction::Type{<:Direction})
    error("Not implemented")
end

function propagate_and_save(p::AbstractOpticalSource{<:Trainable},
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

function backpropagate_with_gradient(
        ∂v, p::AbstractOpticalSource{<:Trainable},
        direction::Type{<:Direction})
    backpropagate_with_gradient!(∂v, p, direction)
end

include("freespace.jl")
export ASProp, RSProp

include("phasemask.jl")
export Phase

include("seeder.jl")
export Seeder

end
