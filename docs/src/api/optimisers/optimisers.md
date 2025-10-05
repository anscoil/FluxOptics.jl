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

### Examples

```julia
using FluxOptics, Optimisers

# Create components
u = ScalarField(ones(ComplexF64, 128, 128), (2.0, 2.0), 1.064)
phase = Phase(u, (x, y) -> 0.0; trainable=true)
mask = Mask(u, (x, y) -> 1.0; trainable=true)
source = ScalarSource(u; trainable=true)

# Per-component rules
rules = make_rules(
    phase => Descent(0.01),
    mask => Momentum(0.05, 0.9),
    source => NoDescent()  # Keep source fixed
)

# Setup with default rule for other parameters
system = source |> phase |> mask
opt_state = setup(rules, Descent(0.001), system)

# Optimization loop
for iter in 1:1000
    grads = gradient(loss, params)[1]
    update!(opt_state, params, grads)
end
```

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

### Examples

```julia
# FISTA for sparse optimization
fista_opt = Fista(0.1)
sparse_rule = ProxRule(fista_opt, IstaProx(0.001, 0.0))

# Standard optimizers
adam_opt = Optimisers.Adam(0.001)
momentum_opt = Momentum(0.01, 0.9)

# No optimization
fixed_opt = NoDescent()
```

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

### Examples

```julia
# Box constraints
clamp_prox = ClampProx(0.0, 1.0)  # Clamp to [0,1]
rule = ProxRule(Descent(0.01), clamp_prox)

# Non-negativity
pos_prox = PositiveProx()
rule = ProxRule(Momentum(0.1, 0.9), pos_prox)

# Soft thresholding (L1 regularization)
ista_prox = IstaProx(0.001, 0.0)  # λ=0.001
rule = ProxRule(Fista(0.05), ista_prox)

# Total variation denoising
tv_prox = TVProx(0.01)  # λ=0.01
rule = ProxRule(Descent(0.1), tv_prox)

# Custom pointwise constraint
custom_prox = PointwiseProx(x -> clamp(x, -π, π))
rule = ProxRule(Descent(0.01), custom_prox)

# Compose multiple constraints
combined = ClampProx(-1.0, 1.0) ∘ PositiveProx()
rule = ProxRule(Descent(0.01), combined)
```

## Optimization Patterns

### Per-Component Learning Rates

```julia
# Fast learning for source, slow for phase
rules = make_rules(
    source => Descent(0.1),    # Fast
    phase => Descent(0.001)    # Slow
)

opt_state = setup(rules, system)
```

### Constrained Phase Optimization

```julia
# Keep phase in [-π, π]
phase_rule = ProxRule(Fista(0.05), ClampProx(-π, π))

rules = make_rules(phase => phase_rule)
opt_state = setup(rules, Descent(0.01), system)
```

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
    source => NoDescent(),                                    # Fixed
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

### Proximal Operators

Proximal operators implement the proximal map:
```
prox_λf(x) = argmin_z { f(z) + (1/2λ)||z - x||² }
```

Common uses:
- **Constraints**: Project onto feasible set (e.g., ClampProx)
- **Regularization**: Encourage desirable properties (e.g., TVProx for smoothness)
- **Sparsity**: Soft thresholding (IstaProx)

### ProxRule Composition

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

### FISTA Acceleration

FISTA provides:
- Faster convergence than gradient descent
- Optimal O(1/k²) convergence rate
- Particularly effective with proximal operators
- Momentum-like behavior without explicit velocity

### Operator Composition

Proximal operators can be composed with `∘`:
```julia
combined = ClampProx(0.0, 1.0) ∘ PositiveProx()
```
Operators applied right-to-left (like function composition).

### Performance

- Proximal operators work in-place when possible
- Pre-allocated buffers avoid allocations
- TV denoising uses efficient iterative algorithm
- GPU-compatible (works with CuArrays)

## Advanced Usage

### Custom Proximal Operator

```julia
using FluxOptics.OptimisersExt.ProximalOperators

struct MyProx <: AbstractProximalOperator
    param::Float64
end

function ProximalOperators.init(prox::MyProx, x::AbstractArray)
    # Return state (can be empty tuple if stateless)
    ()
end

function ProximalOperators.apply!(prox::MyProx, state, x::AbstractArray)
    # Modify x in-place
    @. x = my_projection(x, prox.param)
    x
end

# Use in optimization
rule = ProxRule(Descent(0.01), MyProx(1.0))
```

### Adaptive Learning Rates

```julia
# Start with high learning rate, decrease over time
for epoch in 1:10
    η = 0.1 * 0.9^epoch
    rules = make_rules(phase => Descent(η))
    opt_state = setup(rules, system)
    
    for iter in 1:100
        grads = gradient(loss, params)[1]
        update!(opt_state, params, grads)
    end
end
```

## See Also

- [Metrics](../metrics/index.md) for loss functions
- [Fields](../fields/index.md) for field operations
- [OpticalComponents](../optical_components/index.md) for trainable components
- [Optimisers.jl](https://fluxml.ai/Optimisers.jl/stable/) for more optimization algorithms
