module ProximalOperators

using Optimisers
using ..OptimisersExt: Fista
using LinearAlgebra
export AbstractProximalOperator
export PointwiseProx, IstaProx, ClampProx, PositiveProx, TVProx
export TV_denoise!
export Fista, NoDescent

abstract type AbstractProximalOperator end

function init(prox::AbstractProximalOperator, x::AbstractArray)
    error("Not implemented")
end

function apply!(prox::AbstractProximalOperator, state, x::AbstractArray)
    error("Not implemented")
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
    for (single_prox, state) in zip(reverse(prox.ops), states)
        apply!(single_prox, state, x)
    end
    x
end

include("pointwise_prox.jl")
include("tv_prox.jl")

end
