module ProximalOperators

using Optimisers
using ..OptimisersExt: Fista
using LinearAlgebra
export AbstractProximalOperator, ProxRule
export PointwiseProx, IstaProx, ClampProx, PositiveProx, TVProx
export TV_denoise!
export Fista, NoDescent

"""
    AbstractProximalOperator

Abstract base type for proximal-like operators used in constrained optimization.

In the classical sense, proximal operators implement:
```
prox_λf(x) = argmin_z { f(z) + (1/2λ)||z - x||² }
```

However, this module takes a **pragmatic approach** and includes:
- **Exact proximal operators**: Mathematically rigorous (e.g., soft thresholding)
- **Projections**: Onto constraint sets (e.g., ClampProx, PositiveProx)
- **Plug-and-play operators**: Operators that work well empirically for inverse problems, 
  even without strict proximal interpretation (e.g., learned denoisers, heuristic constraints)

This flexibility follows the spirit of plug-and-play priors in computational imaging, 
where practical performance often matters more than theoretical guarantees.

# Interface

Subtypes must implement:
- `init(prox::AbstractProximalOperator, x::AbstractArray)`: Initialize operator state
- `apply!(prox::AbstractProximalOperator, state, x::AbstractArray)`: Apply operator in-place

# Available Operators

- [`PointwiseProx`](@ref): Apply function element-wise (custom constraints)
- [`ClampProx`](@ref): Box constraints [a, b]
- [`PositiveProx`](@ref): Non-negativity constraint
- [`IstaProx`](@ref): Soft thresholding (exact prox of L1 norm)
- [`TVProx`](@ref): Total variation denoising (approximate prox)

# Composition

Operators can be composed with `∘`:
```julia
combined = ClampProx(0.0, 1.0) ∘ PositiveProx()
```

# Examples

```julia
# Use with ProxRule
prox = ClampProx(-π, π)
rule = ProxRule(Descent(0.01), prox)

# Compose multiple constraints
prox = ClampProx(0.0, 1.0) ∘ PositiveProx()
rule = ProxRule(Momentum(0.1, 0.9), prox)
```

# References

For plug-and-play priors: Venkatakrishnan et al., "Plug-and-Play Priors for Model Based Reconstruction" (2013)

See also: [`ProxRule`](@ref), [`IstaProx`](@ref), [`TVProx`](@ref)
"""
abstract type AbstractProximalOperator end

"""
    ProxRule(rule::AbstractRule, prox::AbstractProximalOperator)

Combine an optimization rule with a proximal operator.

This creates a composite optimization rule that first applies the standard optimization
step, then applies a proximal operator for regularization or constraints. This is
useful for constrained optimization in inverse optics design.

# Arguments
- `rule`: Base optimization rule (e.g., `Descent`, `Momentum`)
- `prox`: Proximal operator to apply after the optimization step

# Returns
`ProxRule` that applies both the optimizer and proximal operator.

# Examples
```jldoctest
julia> prox_descent = ProxRule(Descent(0.01), PositiveProx());  # Positive constraint

julia> tv_regularized = ProxRule(Momentum(0.1, 0.9), TVProx(0.001));  # TV regularization

julia> clamped = ProxRule(Descent(0.05), ClampProx(0.0, 1.0));  # Clamp to [0,1]
```

See also: [`IstaProx`](@ref), [`TVProx`](@ref), [`ClampProx`](@ref), [`Fista`](@ref)
"""
struct ProxRule{R <: AbstractRule, F <: AbstractProximalOperator} <: AbstractRule
    rule::R
    prox::F
end

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
