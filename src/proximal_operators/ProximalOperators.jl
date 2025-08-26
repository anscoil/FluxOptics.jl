module ProximalOperators

export AbstractProximalOperator
export IstaProx, ClampProx, PositiveProx

abstract type AbstractProximalOperator end
abstract type StatelessProximalOperator <: AbstractProximalOperator end

function init(prox::AbstractProximalOperator, x::AbstractArray)
    error("Not implemented")
end

function apply!(prox::AbstractProximalOperator, state, x::AbstractArray)
    error("Not implemented")
end

function init(prox::StatelessProximalOperator, x::AbstractArray)
    ()
end

function get_prox_fun(prox::StatelessProximalOperator)
    error("Not implemented")
end

function apply!(prox::StatelessProximalOperator, state, x::AbstractArray)
    f = get_prox_fun(prox)
    @. x = f(x)
end

struct CompositeProx <: AbstractProximalOperator
    ops::NTuple{N, AbstractProximalOperator} where {N}
end

Base.:∘(a::AbstractProximalOperator, b::AbstractProximalOperator) = CompositeProx((a, b))

Base.:∘(a::AbstractProximalOperator, b::CompositeProx) = CompositeProx((a, b.ops...))

Base.:∘(a::CompositeProx, b::AbstractProximalOperator) = CompositeProx((a.ops..., b))

Base.:∘(a::CompositeProx, b::CompositeProx) = CompositeProx((a.ops..., b.ops...))

function init(prox::CompositeProx, x::AbstractArray)
    reverse(map(op -> init(op, x), prox.ops))
end

function apply!(prox::CompositeProx, states, x::AbstractArray)
    @assert length(states) == length(prox.ops)
    for (single_prox, state) in zip(reverse(prox.ops), reverse(states))
        apply!(single_prox, state, x)
    end
    x
end

include("ista_prox.jl")
include("clamp_prox.jl")
include("positive_prox.jl")

end
