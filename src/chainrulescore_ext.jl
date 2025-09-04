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
) where {P <: AbstractCustomComponent{Trainable{Buffered}}}
    v = propagate_and_save(u, p, direction)

    function pullback(∂v)
        ∂p = get_preallocated_gradient(p)
        u_saved = get_saved_buffer(p)
        ∂u, ∂p = backpropagate_with_gradient(∂v, u_saved, ∂p, p, direction)
        return (NoTangent(), ∂u, Tangent{P}(; ∂p...), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(
        ::typeof(propagate), u, p::P,
        direction::Type{<:Direction}
) where {P <: AbstractCustomComponent{Trainable{Unbuffered}}}
    u_saved = alloc_saved_buffer(u, p)
    v = propagate_and_save(u, u_saved, p, direction)

    function pullback(∂v)
        ∂p = alloc_gradient(p)
        ∂u, ∂p = backpropagate_with_gradient(∂v, u_saved, ∂p, p, direction)
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
) where {P <: AbstractCustomComponent{Trainable{Buffered}}}
    v = propagate_and_save!(u, p, direction)

    function pullback(∂v)
        ∂p = get_preallocated_gradient(p)
        u_saved = get_saved_buffer(p)
        ∂u, ∂p = backpropagate_with_gradient!(∂v, u_saved, ∂p, p, direction)
        return (NoTangent(), ∂u, Tangent{P}(; ∂p...), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(
        ::typeof(propagate!), u, p::P,
        direction::Type{<:Direction}
) where {P <: AbstractCustomComponent{Trainable{Unbuffered}}}
    u_saved = alloc_saved_buffer(u, p)
    v = propagate_and_save!(u, u_saved, p, direction)

    function pullback(∂v)
        ∂p = alloc_gradient(p)
        ∂u, ∂p = backpropagate_with_gradient!(∂v, u_saved, ∂p, p, direction)
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
) where {P <: AbstractCustomSource{Trainable{Buffered}}}
    v = propagate_and_save(p, direction)

    function pullback(∂v)
        ∂p = get_preallocated_gradient(p)
        ∂p = backpropagate_with_gradient(∂v, ∂p, p, direction)
        return (NoTangent(), Tangent{P}(; ∂p...), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(propagate), p::P,
        direction::Type{<:Direction}
) where {P <: AbstractCustomSource{Trainable{Unbuffered}}}
    v = propagate_and_save(p, direction)

    function pullback(∂v)
        ∂p = alloc_gradient(p)
        ∂p = backpropagate_with_gradient(∂v, ∂p, p, direction)
        return (NoTangent(), Tangent{P}(; ∂p...), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(::Type{<:ScalarField}, data, ds, lambdas, lambdas_collection)
    y = ScalarField(data, ds, lambdas, lambdas_collection)
    function pullback(∂y)
        (NoTangent(), ∂y.data, NoTangent(), NoTangent(), NoTangent())
    end
    return y, pullback
end

function ChainRulesCore.ProjectTo(u::ScalarField)
    function pullback(∂y)
        if ∂y.data isa NoTangent
            NoTangent()
        else
            ScalarField(∂y.data, u.ds, u.lambdas, u.lambdas_collection)
        end
    end
    pullback
end
