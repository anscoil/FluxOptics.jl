# Modes API

## Gaussian Beam Modes

### Gaussian Beams

```@docs
Gaussian1D
Gaussian
```

### Hermite-Gaussian Modes

```@docs
HermiteGaussian1D
HermiteGaussian
hermite_gaussian_groups
```

### Laguerre-Gaussian Modes

```@docs
LaguerreGaussian
```

## Spatial Layouts

### Layout Types

```@docs
Modes.PointLayout
Modes.GridLayout
Modes.TriangleLayout
Modes.CustomLayout
```

### Mode Stack Generation

```@docs
generate_mode_stack
```

## Speckle Generation

```@docs
generate_speckle
```

## Technical Notes

### Mode Normalization
- All modes are normalized to unit power by default
- Power = ∫∫ |u(x,y)|² dx dy = 1
- For custom normalization, use `norm_constant` parameter

### Mode Propagation
- Use `λ` and `z` parameters for Gaussian beam propagation
- Includes Gouy phase shift
- `constant_phase=true` includes exp(ikz) term

### Layout Centering
- Layouts are centered at (0, 0) by default
- Use coordinate transformations to shift layouts

### Speckle Statistics
- Correlation length ≈ λ / NA
- High NA → fine speckle (small correlation length)
- Low NA → coarse speckle (large correlation length)
- Speckle is fully developed (unit contrast) without envelope

## See Also

- [GridUtils](../gridutils/index.md) for coordinate systems
- [Fields](../fields/index.md) for ScalarField operations
