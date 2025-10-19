# GridUtils

Coordinate systems and transformations for optical field grids.

## Overview

The `GridUtils` module provides:
- **Spatial coordinate generation** for optical field grids
- **2D coordinate transformations** (translations, rotations)
- **Coordinate composition** for complex geometries

## Quick Example

```@example gridutils
using FluxOptics
using Statistics

# Generate 2D coordinate grid
xv, yv = spatial_vectors((128, 128), (2.0, 2.0); offset=(-5.0, 30.5))

(mean(xv), mean(yv))  # Grid center = -offset
```

```@example gridutils
# Create transformation: shift then rotate
transform = Rot2D(π/4) ∘ Shift2D(10.0, 5.0)

# Apply to a point
point = transform([0.0, 0.0])
```

## Key Functions

- [`spatial_vectors`](@ref): Generate coordinate arrays
- [`Shift2D`](@ref): 2D translation transformation
- [`Rot2D`](@ref): 2D rotation transformation
- [`Id2D`](@ref): Identity transformation

## See Also

- [Modes](../modes/index.md) for using coordinates with optical modes
- [Fields](../fields/index.md) for ScalarField grid structure

## Index

```@index
Modules = [FluxOptics.GridUtils]
Order = [:type, :function]
```
