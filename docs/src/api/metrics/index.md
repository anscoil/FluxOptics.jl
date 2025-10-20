# Metrics

Loss functions for inverse design optimization.

## Overview

The `Metrics` module provides:
- **Field overlap**: `DotProduct` and `PowerCoupling` metrics
- **Field matching**: `SquaredFieldDifference` for complex field matching
- **Intensity matching**: `SquaredIntensityDifference` for intensity-based objectives
- **AD-compatible**: Automatic differentiation support via custom gradients

## Examples

### Power Coupling

```@example metrics1
using FluxOptics

# Create fields
xv, yv = spatial_vectors(128, 128, 2.0, 2.0)
u_current = ScalarField(Gaussian(15.0)(xv, yv), (2.0, 2.0), 1.064)
u_target = ScalarField(Gaussian(20.0)(xv, yv), (2.0, 2.0), 1.064)

# Power coupling metric
metric = PowerCoupling(u_target)

# Evaluate coupling (returns power in Watts)
power_coupled = metric(u_current)
power_coupled[]
```

### Field Matching

```@example metrics2
using FluxOptics

# Create fields with different phases
xv, yv = spatial_vectors(128, 128, 2.0, 2.0)
gaussian = Gaussian(15.0)(xv, yv)
u_current = ScalarField(gaussian, (2.0, 2.0), 1.064)
u_target = ScalarField(gaussian .* cis.(0.01 .* (xv.^2 .+ yv'.^2)), (2.0, 2.0), 1.064)

# Squared field difference
metric = SquaredFieldDifference(u_target)
loss = metric(u_current)
loss[]
```

### Intensity Matching

```@example metrics3
using FluxOptics

# Create field and target intensity pattern
xv, yv = spatial_vectors(128, 128, 2.0, 2.0)
u_current = ScalarField(Gaussian(35.0)(xv, yv), (2.0, 2.0), 1.064)

# Target: ring pattern
r = sqrt.(xv.^2 .+ yv'.^2)
target_intensity = exp.(-(r .- 30).^2 / 100)
target_intensity ./= sum(target_intensity) * 2.0 * 2.0

# Intensity matching metric
metric = SquaredIntensityDifference((u_current, target_intensity))
loss = metric(u_current)
loss[]
```

## Key Types

- [`DotProduct`](@ref): Complex overlap integral ⟨u,v⟩
- [`PowerCoupling`](@ref): Power coupled to target mode(s)
- [`SquaredFieldDifference`](@ref): |u - v|² field matching
- [`SquaredIntensityDifference`](@ref): |I_u - I_v|² intensity matching

## Key Functions

- [`compute_metric`](@ref Metrics.compute_metric): Evaluate metric on fields
- [`backpropagate_metric`](@ref Metrics.backpropagate_metric): Gradient computation (internal)

## See Also

- [Typical Workflow](../index.md#typical-workflow-beam-splitter) - Complete example of building and optimizing an optical system
- [OptimisersExt](../optimisers/index.md) for optimization algorithms
- [System](../optical_components/system/index.md) for building systems to optimize

## Index

```@index
Modules = [FluxOptics.Metrics]
Order = [:type, :function]
```
