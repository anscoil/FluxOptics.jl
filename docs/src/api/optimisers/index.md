# OptimisersExt

Optimization rules and proximal operators for constrained inverse design.

## Overview

The `OptimisersExt` module extends `Optimisers.jl` with:
- **Custom optimization rules**: FISTA, NoDescent
- **Proximal operators**: Constraints and regularization (TV, sparsity, clamping)
- **Per-parameter rules**: Different optimizers for different components
- **Proximal-gradient methods**: Combine optimization with constraints

## Quick Example

```@example
using FluxOptics

# Define source
xv, yv = spatial_vectors(128, 128, 1.0, 1.0)
u = ScalarField(Gaussian(20.0)(xv, yv), (1.0, 1.0), 1.064)
source = ScalarSource(u)

# Define optical components
phasemask = Phase(u, (x, y) -> 0.0; trainable=true)
mask = FourierMask(u, (fx, fy) -> 1.0; trainable=true)

# Define optical system
system = source |> phasemask |> mask

# Per-component optimization rules
rules = make_rules(
    phasemask => ProxRule(Descent(0.01), ClampProx(-π, π)),  # Constrained phase
    mask => Momentum(0.1, 0.9)                               # Momentum for mask
)

# Setup
opt_state = setup(rules, system)
```

## Key Types

- [`ProxRule`](@ref): Combine optimizer with proximal operator
- [`Fista`](@ref): Fast iterative shrinkage-thresholding
- [`NoDescent`](@ref): No-op optimizer for fixed parameters

## Proximal Operators

- [`IstaProx`](@ref): Soft thresholding for sparsity
- [`TVProx`](@ref): Total variation regularization
- [`ClampProx`](@ref): Box constraints
- [`PositiveProx`](@ref): Non-negativity constraint
- [`PointwiseProx`](@ref): Custom element-wise constraints

## Key Functions

- [`make_rules`](@ref): Create per-parameter optimization rules
- [`setup`](@ref FluxOptics.setup(::IdDict)): Initialize optimization state with custom rules
- `update!`: Apply optimization step (from [Optimisers.jl](https://fluxml.ai/Optimisers.jl/stable/api/#Optimisers.update!))

## See Also

- [Typical Workflow](../index.md#typical-workflow-beam-splitter) - Complete example of building and optimizing an optical system
- [Metrics](../metrics/index.md) for loss functions
- [Optimisers.jl](https://fluxml.ai/Optimisers.jl/stable/) for base optimization algorithms

## Index

```@index
Modules = [FluxOptics.OptimisersExt, FluxOptics.OptimisersExt.ProximalOperators]
Order = [:type, :function]
```
