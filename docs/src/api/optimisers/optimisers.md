# OptimisersExt API

```@meta
CurrentModule = FluxOptics.OptimisersExt
```

## Setup and Update

```@docs
make_rules
Optimisers.setup(::IdDict)
```

**Note:** `update!(opt_state, params, grads)` is re-exported from [Optimisers.jl](https://fluxml.ai/Optimisers.jl/stable/api/#Optimisers.update!). It applies the optimization step to update parameters. See their documentation for details.

## Optimization Rules

### Custom Rules

```@docs
Fista
NoDescent
ProxRule
```

### From Optimisers.jl

FluxOptics also exports standard rules from Optimisers.jl:
- `Descent(η)`: Gradient descent with learning rate η
- `Momentum(η, ρ)`: Momentum optimizer
- `Nesterov(η, ρ)`: Nesterov accelerated gradient

See [Optimisers.jl documentation](https://fluxml.ai/Optimisers.jl/stable/) for more rules (Adam, AdaGrad, etc.).

## Proximal Operators

### Abstract Type

```@docs
AbstractProximalOperator
```

### Pointwise Operators

```@docs
PointwiseProx
ClampProx
PositiveProx
```

### Sparsity and Regularization

```@docs
IstaProx
TVProx
TV_denoise!
```

## Optimization Patterns

### Sparse Phase Mask Design

```julia
# Encourage sparsity with soft thresholding
sparse_rule = ProxRule(
    Fista(0.01),
    IstaProx(0.001, 0.0)  # L1 penalty
)

rules = make_rules(phase => sparse_rule)
opt_state = setup(rules, system)
```

### Smooth Phase with TV Regularization

```julia
# Total variation for smooth phase profiles
tv_rule = ProxRule(
    Momentum(0.05, 0.9),
    TVProx(0.01)  # Smoothness penalty
)

rules = make_rules(phase => tv_rule)
opt_state = setup(rules, system)
```

### Mixed Optimization Strategy

```julia
# Different strategies for different components
rules = make_rules(
    phase => ProxRule(Fista(0.05), ClampProx(-π, π)),       # Constrained
    mask => ProxRule(Momentum(0.1, 0.9), PositiveProx())    # Positive only
)

opt_state = setup(rules, Descent(0.001), system)
```

### Freezing Specific Components

```julia
# Keep some components fixed during optimization
rules = make_rules(
    component_to_freeze => NoDescent()
)

# Others use default rule
opt_state = setup(rules, Descent(0.01), system)
```

**Note:** `NoDescent()` prevents parameter updates but still computes gradients. For better performance when permanently freezing components, create them with `trainable=false` at construction. However, this requires rebuilding the system to change trainability.

## Technical Notes

### ProxRule

`ProxRule` combines optimization with proximal operators:
1. Apply gradient step: `x_temp = x - η∇f(x)`
2. Apply proximal operator: `x_new = prox(x_temp)`

This enables constrained optimization while maintaining differentiability.

### make_rules Behavior

- Maps components/arrays to optimization rules
- Extracts trainable arrays from components automatically
- Returns `IdDict{AbstractArray, AbstractRule}`
- Compatible with nested structures (OpticalSystem, sequences)

### setup with Rules Dictionary

- Parameters in rules dict use specified rules
- Other parameters use default rule (or NoDescent if no default)
- Warning if no trainable parameters found

### Operator Composition

Proximal operators can be composed with `∘`:
```julia
combined = ClampProx(0.0, 1.0) ∘ PositiveProx()
```
Operators applied right-to-left (like function composition). The resulting operator may not be a rigorous proximal operator but can still be used as a Plug and Play prior.

### Performance

- Proximal operators work in-place when possible
- Pre-allocated buffers avoid allocations
- TV denoising uses efficient iterative algorithm
- GPU-compatible (works with CuArrays)

## See Also

- [Metrics](../metrics/index.md) for loss functions
- [Fields](../fields/index.md) for field operations
- [OpticalComponents](../optical_components/index.md) for trainable components
- [Optimisers.jl](https://fluxml.ai/Optimisers.jl/stable/) for more optimization algorithms
