import Optimisers
using Optimisers: AbstractRule, mapvalue, _trainable, isnumeric, subtract!, Leaf

Optimisers.trainable(p::OpticalChain) = (; layers = p.layers)

Optimisers.trainable(p::AbstractOpticalComponent) = OpticalComponents.trainable(p)

function Optimisers.setup(
        rules::IdDict{K, <:AbstractRule}, default_rule::AbstractRule, model
) where {K}
    cache = IdDict()
    tree = Optimisers._setup(rules, default_rule, model; cache)
    isempty(cache) && @warn "setup found no trainable parameters in this model"
    tree
end

function Optimisers.setup(rules::IdDict{K, <:AbstractRule}, model) where {K}
    Optimisers.setup(rules, NoDescent(), model)
end

function Optimisers._setup(rules, default_rule, x; cache)
    haskey(cache, x) && return cache[x]
    if isnumeric(x)
        rule = haskey(rules, x) ? rules[x] : default_rule
        ℓ = Leaf(rule, Optimisers.init(rule, x))
        if isbits(x)
            cache[nothing] = nothing  # just to disable the warning
            ℓ
        else
            cache[x] = ℓ
        end
    else
        mapvalue(xᵢ -> Optimisers._setup(rules, default_rule, xᵢ; cache), _trainable(x))
    end
end

function make_rules(pairs::Pair{
        <:K, <:AbstractRule}...) where {
        K <: Union{AbstractArray, AbstractOpticalComponent}}
    pairs = map(
        ((x, v),) -> isa(x, AbstractOpticalComponent) ? (get_data(x), v) : (x, v), pairs)
    IdDict{AbstractArray, AbstractRule}(pairs)
end

struct ProxRule{R <: AbstractRule, F <: AbstractProximalOperator} <: AbstractRule
    rule::R
    prox::F
end

function Optimisers.init(o::ProxRule, x::AbstractArray)
    Optimisers.init(o.rule, x), ProximalOperators.init(o.prox, x)
end

function Optimisers.apply!(o::ProxRule, state, x, x̄)
    opt_state, prox_state = state
    opt_state, x̄′ = Optimisers.apply!(o.rule, opt_state, x, x̄)
    state = opt_state, prox_state
    state, x̄′
end

function Optimisers._update!(ℓ::Leaf{<:ProxRule, S}, x; grads, params) where {S}
    haskey(params, (ℓ, x)) && return params[(ℓ, x)]
    ℓ.frozen && return x
    params[(ℓ, x)] = if haskey(grads, ℓ)
        ℓ.state, x̄′ = Optimisers.apply!(ℓ.rule, ℓ.state, x, grads[ℓ]...)
        subtract!(x, x̄′)
        _, prox_state = ℓ.state
        ProximalOperators.apply!(ℓ.rule.prox, prox_state, x)
    else
        x # no gradient seen
    end
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
