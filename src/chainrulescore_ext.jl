using ChainRulesCore

function ChainRulesCore.rrule(
        ::typeof(propagate), u, p::AbstractCustomComponent{Static},
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
) where {P <: AbstractCustomComponent{<:Trainable}}
    v = propagate_and_save(u, p, direction)

    function pullback(∂v)
        ∂u, ∂p = backpropagate_with_gradient(∂v, p, direction)
        return (NoTangent(), ∂u, Tangent{P}(; ∂p...), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(
        ::typeof(propagate!), u, p::AbstractCustomComponent{Static},
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
) where {P <: AbstractCustomComponent{<:Trainable}}
    v = propagate_and_save!(u, p, direction)
    function pullback(∂v)
        ∂u, ∂p = backpropagate_with_gradient!(∂v, p, direction)
        return (NoTangent(), ∂u, Tangent{P}(; ∂p...), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(
        ::typeof(propagate), p::AbstractCustomSource{Static},
        direction::Type{<:Direction})
    v = propagate(p, direction)

    function pullback(∂v)
        return (NoTangent(), NoTangent(), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(propagate), p::P,
        direction::Type{<:Direction}
) where {P <: AbstractCustomSource{<:Trainable}}
    v = propagate_and_save(p, direction)

    function pullback(∂v)
        ∂p = backpropagate_with_gradient(∂v, p, direction)
        return (NoTangent(), Tangent{P}(; ∂p...), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(::Type{<:ScalarField}, data, lambdas, lambdas_collection)
    y = ScalarField(data, lambdas, lambdas_collection)
    function pullback(∂y)
        (NoTangent(), ∂y.data, NoTangent(), NoTangent())
    end
    return y, pullback
end

function ChainRulesCore.ProjectTo(u::ScalarField)
    function pullback(∂y)
        if ∂y.data isa NoTangent
            NoTangent()
        else
            ScalarField(∂y.data, u.lambdas, u.lambdas_collection)
        end
    end
    pullback
end
