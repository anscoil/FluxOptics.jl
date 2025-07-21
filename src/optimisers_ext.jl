using Optimisers
import Optimisers: init, mapvalue, _trainable, isnumeric, apply!, subtract!, Leaf

function Optimisers.setup(
        rules::IdDict{K, <:AbstractRule}, default_rule::AbstractRule, model
) where {K}
    cache = IdDict()
    tree = Optimisers._setup(rules, default_rule, model; cache)
    isempty(cache) && @warn "setup found no trainable parameters in this model"
    tree
end

function Optimisers._setup(rules, default_rule, x; cache)
    haskey(cache, x) && return cache[x]
    if isnumeric(x)
        rule = haskey(rules, x) ? rules[x] : default_rule
        ℓ = Leaf(rule, init(rule, x))
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

function rules_dict(pairs::Pair{K, <:AbstractRule}...) where {K}
    return IdDict{Any, AbstractRule}(pairs)
end

struct ProxRule{R <: AbstractRule} <: AbstractRule
    rule::R
    prox::Function
end

function Optimisers.apply!(o::ProxRule, state, x, x̄)
    return apply!(o.rule, state, x, x̄)
end

Optimisers.init(o::ProxRule, x::AbstractArray) = init(o.rule, x)

function Optimisers._update!(ℓ::Leaf{<:ProxRule, S}, x; grads, params) where {S}
    haskey(params, (ℓ, x)) && return params[(ℓ, x)]
    ℓ.frozen && return x
    params[(ℓ, x)] = if haskey(grads, ℓ)
        ℓ.state, x̄′ = apply!(ℓ.rule, ℓ.state, x, grads[ℓ]...)
        subtract!(x, x̄′)
        ℓ.rule.prox(x)
    else
        x # no gradient seen
    end
end

struct Fista <: AbstractRule
    eta::Any
end

function init(o::Fista, x::AbstractArray{T}) where {T}
    (T(1), copy(x), zero(x))
end

function apply!(o::Fista, (tk, xk, newdx), x::AbstractArray{T}, dx) where {T}
    η = T(o.eta)
    tkn = (1+sqrt(1+4*tk^2))/2
    β = (tk-1)/tkn

    @. newdx = η*dx - β*(x-xk)
    copyto!(xk, x)

    (tkn, xk, newdx), newdx
end
