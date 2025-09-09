module ProximalOperators

using ..Fields
using LinearAlgebra
using Optimisers
export AbstractProximalOperator
export PointwiseProx, IstaProx, ClampProx, PositiveProx, TVProx
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
    for (single_prox, state) in zip(reverse(prox.ops), reverse(states))
        apply!(single_prox, state, x)
    end
    x
end

struct Fista <: AbstractRule
    eta::Real

    function Fista(eta)
        new(eta^2)
    end
end

function Optimisers.init(o::Fista, x::AbstractArray{T}) where {T}
    (T(1), copy(x), zero(x))
end

function Optimisers.apply!(o::Fista, (tk, xk, newdx), x::AbstractArray{T}, dx) where {T}
    η = T(o.eta)
    tkn = (1+sqrt(1+4*tk^2))/2
    β = (tk-1)/tkn

    @. newdx = η*dx - β*(x-xk)
    copyto!(xk, x)

    (tkn, xk, newdx), newdx
end

struct NoDescent <: AbstractRule end

Optimisers.init(o::NoDescent, x) = ()

Optimisers.apply!(o::NoDescent, _, x, dx) = ((), 0)

include("pointwise_prox.jl")
include("tv_prox.jl")

end
