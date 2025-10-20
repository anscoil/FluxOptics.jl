# Metrics API

```@meta
CurrentModule = FluxOptics.Metrics
```

## Abstract Types

```@docs
AbstractMetric
```

## Core Functions

```@docs
(AbstractMetric)(::ScalarField)
compute_metric
backpropagate_metric
```

## Dot Product Metrics

```@docs
DotProduct
```

```@docs
PowerCoupling
```

## Field Difference Metrics

```@docs
SquaredFieldDifference
```

```@docs
SquaredIntensityDifference
```

## Technical Notes

### Metric Types
- All metrics are subtypes of `AbstractMetric`
- Metrics are callable: `metric(field)` calls `compute_metric`
- Support single or multiple field arguments

### Mode Selectivity
- `mode_selective=true`: Compute per-mode metrics (diagonal)
- `mode_selective=false`: Compute full coupling matrix
- Affects output dimensionality and computational cost

### Normalization
- `PowerCoupling` normalizes targets internally (copies first)
- `DotProduct` uses fields as-is
- Consider normalizing input fields before metrics

### Performance
- Metrics cache internal buffers to avoid allocations
- Use in-place operations where possible

## See Also

- [OptimisersExt](../optimisers/index.md) for optimization algorithms
