# Bulk Propagators API

```@meta
CurrentModule = FluxOptics.OpticalComponents
```

## Types

```@docs
AS_BPM
Shift_BPM
```

## Technical Notes

### Split-Step Method

AS_BPM uses split-step Fourier method:
1. Apply phase shift from Δn: `exp(i k₀ Δn dz cos(θ))`
2. Propagate dz in background medium (n₀)
3. Repeat for all slices

**Note:** Implementation adds half-steps (dz/2) before the first and after the last phase mask for symmetry, though this has minimal impact on results.

Cosine correction accounts for oblique propagation in tilted beams.

### Slice Discretization

More slices → better accuracy but slower:
- Rule of thumb: `dz < λ / (2 max|Δn|)`
- For weak variations: 10-50 slices often sufficient
- For strong gradients: 100-1000 slices may be needed

## See Also

- [ASProp](@ref) for homogeneous propagation
- [Core](../core/index.md) for trainability
