module OpticalComponents

using Functors
using ..Types

export make_fft_plans
export propagate, propagate!, backpropagate, backpropagate!

abstract type AbstractPropagator{M} <: AbstractOpticalComponent{M} end

trainable(p::AbstractOpticalComponent{Static}) = NamedTuple{}()

function trainable(p::AbstractOpticalComponent{Trainable})
    error("Not implemented")
end

function propagate!(u, args...)
    error("Not implemented")
end

function propagate(u, args...)
    propagate!(copy(u), args...)
end

function propagate_and_save!(u, p::AbstractOpticalComponent{Trainable}, args...)
    error("Not implemented")
end

function propagate_and_save(u, p::AbstractOpticalComponent{Trainable}, args...)
    propagate!(copy(u), p, args...)
end

function backpropagate!(u, args...)
    error("Not implemented")
end

function backpropagate(u, args...)
    backpropagate!(copy(u), args...)
end

function backpropagate_with_gradients!(
        ∂v, ∂p::NamedTuple, p::AbstractOpticalComponent{Trainable}, args...)
    error("Not implemented")
end

function backpropagate_with_gradients(∂v, p::AbstractOpticalComponent{Trainable}, args...)
    ∂p = fmap(similar, trainable(p))
    backpropagate_with_gradients!(copy(∂v), ∂p, p, args...)
end

include("freespace.jl")
export ASProp, RSProp

include("phasemask.jl")
export Phase

end
