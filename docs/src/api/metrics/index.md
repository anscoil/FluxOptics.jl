# Metrics

Metric types for inverse design optimization.

## Overview

The `Metrics` module provides:
- **Metric types**: Object-oriented approach to optimization objectives
- **Field overlap**: `DotProduct` and `PowerCoupling` metrics
- **Field matching**: `SquaredFieldDifference` for complex field matching
- **Intensity matching**: `SquaredIntensityDifference` for intensity-based objectives
- **AD-compatible**: Automatic differentiation support via custom gradients

## Quick Example

```julia
using FluxOptics

# Create target field
target = ScalarField(target_data, (2.0, 2.0), 1.064)

# Power coupling metric
metric = PowerCoupling(target)

# Evaluate on current field
u_current = system()
loss = metric(u_current)  # Returns power coupled to target

# Use in optimization
function objective()
    u = system()
    1.0 - metric(u)[]  # Maximize coupling
end
```

## Key Types

- [`DotProduct`](@ref): Complex overlap integral ⟨u,v⟩
- [`PowerCoupling`](@ref): Power coupled to target mode(s)
- [`SquaredFieldDifference`](@ref): |u - v|² field matching
- [`SquaredIntensityDifference`](@ref): |I_u - I_v|² intensity matching

## Key Functions

- [`(AbstractMetric)`](@ref): Callable interface (preferred) - evaluate metric
- [`FluxOptics.Metrics.compute_metric`](@ref): Explicit evaluation function
- [`FluxOptics.Metrics.backpropagate_metric`](@ref): Gradient computation (internal)

## See Also

- [OptimisersExt](../optimisers/index.md) for optimization algorithms

## Index

```@index
Modules = [FluxOptics.Metrics]
Order = [:type, :function]
```
