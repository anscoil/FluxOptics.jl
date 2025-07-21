using ChainRulesCore

function ChainRulesCore.rrule(
        ::typeof(propagate), u, p::AbstractOpticalComponent{Static},
        direction::Type{<:Direction})
    v = propagate(u, p, direction)

    function pullback(∂v)
        ∂u = backpropagate(∂v, p, direction)
        return (NoTangent(), ∂u, NoTangent(), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(
        ::typeof(propagate), u, p::P,
        direction::Type{<:Direction}
) where {P <: AbstractOpticalComponent{<:Trainable}}
    v = propagate_and_save(u, p, direction)

    function pullback(∂v)
        ∂u, ∂p = backpropagate_with_gradient(∂v, p, direction)
        return (NoTangent(), ∂u, Tangent{P}(; ∂p...), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(
        ::typeof(propagate!), u, p::AbstractOpticalComponent{Static},
        direction::Type{<:Direction})
    v = propagate!(u, p, direction)

    function pullback(∂v)
        ∂u = backpropagate!(∂v, p, direction)
        return (NoTangent(), ∂u, NoTangent(), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(
        ::typeof(propagate!), u, p::P,
        direction::Type{<:Direction}
) where {P <: AbstractOpticalComponent{<:Trainable}}
    v = propagate_and_save!(u, p, direction)
    function pullback(∂v)
        ∂u, ∂p = backpropagate_with_gradient!(∂v, p, direction)
        return (NoTangent(), ∂u, Tangent{P}(; ∂p...), NoTangent())
    end

    return v, pullback
end
