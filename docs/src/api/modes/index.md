# Modes

Optical mode generation and spatial layouts.

## Overview

The `Modes` module provides:
- **Gaussian beam families** (Gaussian, Hermite-Gaussian, Laguerre-Gaussian)
- **Spatial layouts** for mode composition and multi-mode configurations
- **Speckle generation** with controlled statistics and envelope
- **Mode stacks** for multi-mode generation

## Quick Example

```julia
using FluxOptics

# Single Gaussian mode
gaussian = Gaussian(20.0)  # 20 μm waist
xv, yv = spatial_vectors(128, 128, 2.0, 2.0)
field = gaussian(xv, yv)

# Array of Hermite-Gaussian modes
modes = hermite_gaussian_groups(15.0, 4)  # All HG modes up to order 3

# Spatial layout with replicated modes
layout = Modes.GridLayout(3, 3, 50.0, 50.0)  # 3×3 grid
mode_stack = generate_mode_stack(layout, 128, 128, 2.0, 2.0, gaussian)
```

## Key Types

- [`Gaussian`](@ref), [`HermiteGaussian`](@ref), [`LaguerreGaussian`](@ref): Beam modes
- [`PointLayout`](@ref Modes.PointLayout), [`GridLayout`](@ref Modes.GridLayout), [`TriangleLayout`](@ref Modes.TriangleLayout): Spatial arrangements
- [`CustomLayout`](@ref Modes.CustomLayout): User-defined layouts

## Key Functions

- [`hermite_gaussian_groups`](@ref): Generate complete mode sets
- [`generate_mode_stack`](@ref): Create multi-mode field arrays
- [`generate_speckle`](@ref): Random speckle patterns

## See Also

- [GridUtils](../gridutils/index.md) for coordinate transformations
- [Fields](../fields/index.md) for using modes with ScalarField

## Index

```@index
Modules = [FluxOptics.Modes]
Order = [:type, :function]
```
