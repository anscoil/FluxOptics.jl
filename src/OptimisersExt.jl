module OptimisersExt

using ..OpticalComponents
using Optimisers
using Optimisers: mapvalue, _trainable, isnumeric, subtract!, Leaf
using Functors

export make_rules, setup, update!
export Descent, Momentum, Nesterov, Fista, NoDescent
export ProximalOperators

"""
    Fista(η)

Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) optimizer.

FISTA is an accelerated gradient method particularly effective for problems with
proximal operators. It provides faster convergence than standard gradient descent
for sparse and constrained optimization problems common in computational optics.

# Arguments
- `η`: Learning rate (stored as η²)

# Examples
```jldoctest
julia> fista_opt = Fista(0.1);

julia> sparse_rule = ProxRule(Fista(0.05), IstaProx(0.001, 0.0));  # Sparse optimization
```

See also: [`ProxRule`](@ref), [`IstaProx`](@ref), [`TVProx`](@ref)
"""
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

"""
    NoDescent()

No-operation optimizer that performs no parameter updates.

Useful as a default rule for parameters that should remain fixed during optimization,
or for ablation studies where you want to disable optimization for certain components.

# Examples
```julia
rules = make_rules(
           fixed_component => NoDescent(),    # Don't optimize this
           active_component => Descent(0.01)  # Optimize this one
       );
```

See also: [`make_rules`](@ref), [`setup`](@ref Optimisers.setup(::IdDict)), [`ProxRule`](@ref)
"""
struct NoDescent <: AbstractRule end

Optimisers.init(o::NoDescent, x) = ()

Optimisers.apply!(o::NoDescent, _, x, dx) = ((), 0)

Optimisers.trainable(p::OpticalSystem) = (; source = p.source, components = p.components)

Optimisers.trainable(p::AbstractOpticalComponent) = OpticalComponents.trainable(p)

"""
    setup(rules::IdDict{<:AbstractArray, <:AbstractRule}, model)
    setup(rules::IdDict{<:AbstractArray, <:AbstractRule}, default_rule::AbstractRule, model)

Set up optimization state with different rules for different parameters.

This extends `Optimisers.setup` to accept a dictionary of rules, allowing fine-grained
control over which optimizer is used for which parameters. Parameters not in the
rules dictionary use the default rule (or `NoDescent()` if no default provided).

# Arguments
- `rules`: Dictionary mapping parameter arrays to optimization rules
- `default_rule`: Default rule for parameters not in the rules dictionary
- `model`: The model/parameters to optimize

# Returns
Nested optimization state structure matching the model structure.

# Examples
```jldoctest
julia> u = ScalarField(zeros(ComplexF64, 32, 32), (2.0, 2.0), 1.064);

julia> phase_mask = Phase(u, (x, y) -> 0.0; trainable=true);

julia> mask = Mask(u, (x, y) -> 1.0; trainable=true);

julia> rules = make_rules(
           phase_mask => Descent(0.01),
           mask => Momentum(0.1, 0.9)
       );

julia> u = ScalarField(zeros(ComplexF64, 32, 32), (2.0, 2.0), 1.064);

julia> source = ScalarSource(u; trainable=true);

julia> opt_state = setup(rules, NoDescent(), source |> phase_mask |> mask);
```

See also: [`make_rules`](@ref), `Optimisers.update!`, [`ProxRule`](@ref), [`NoDescent`](@ref)
"""
function Optimisers.setup(rules::IdDict{K, <:AbstractRule}, model) where {K}
    Optimisers.setup(rules, NoDescent(), model)
end

function Optimisers.setup(rules::IdDict{K, <:AbstractRule},
                          default_rule::AbstractRule,
                          model) where {K}
    cache = IdDict()
    tree = Optimisers._setup(rules, default_rule, model; cache)
    isempty(cache) && @warn "setup found no trainable parameters in this model"
    tree
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

"""
    make_rules(pairs::Pair{<:AbstractArray, <:AbstractRule}...)
    make_rules(pairs::Pair{<:AbstractOpticalComponent, <:AbstractRule}...)

Create a dictionary of optimization rules for specific parameters or optical components.

This function creates an `IdDict` that maps parameter arrays to their corresponding
optimization rules. This allows different parts of the optical system to use different
optimizers (e.g., different learning rates, different algorithms).

# Arguments
- `pairs`: Pairs of (parameter/component, rule) where:
  - First element: `AbstractArray` (parameter) or `AbstractOpticalComponent` 
  - Second element: `AbstractRule` (optimization rule like `Descent(0.01)`)

# Returns
`IdDict{AbstractArray, AbstractRule}` mapping parameter arrays to optimization rules.

# Examples
```jldoctest
julia> u = ScalarField(zeros(ComplexF64, 64, 64), (2.0, 2.0), 1.064);

julia> phase_mask = Phase(u, (x, y) -> 0.1*(x^2 + y^2); trainable=true);

julia> source = ScalarSource(u; trainable=true);

julia> rules = make_rules(
           phase_mask => Descent(0.01),    # Slow learning for phase
           source => Descent(0.1)     # Fast learning for source
       );

julia> typeof(rules)
IdDict{AbstractArray, Optimisers.AbstractRule}

julia> length(rules)
2

julia> opt_state = setup(rules, source |> phase_mask);
```

See also: [`setup`](@ref Optimisers.setup(::IdDict)), [`ProxRule`](@ref), [`Phase`](@ref), [`ScalarSource`](@ref)
"""
function make_rules(pairs::Pair{<:K, <:AbstractRule}...) where {K <: Union{AbstractArray,
                                                                      AbstractOpticalComponent}}
    new_pairs = Vector{Tuple{AbstractArray, AbstractRule}}([])
    for (x, v) in pairs
        if isa(x, AbstractOpticalComponent)
            data = get_data(x)
            if isa(data, Tuple)
                foreach(d -> isa(d, AbstractArray) ? push!(new_pairs, (d, v)) : nothing,
                        data)
            end
            if isa(data, AbstractArray)
                push!(new_pairs, (data, v))
            end
        end
        if isa(x, AbstractArray)
            push!(new_pairs, (x, v))
        end
    end
    IdDict{AbstractArray, AbstractRule}(new_pairs)
end

include("proximal_operators/ProximalOperators.jl")
using .ProximalOperators
export AbstractProximalOperator, ProxRule
export PointwiseProx, IstaProx, ClampProx, PositiveProx, TVProx
export TV_denoise!

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

end
