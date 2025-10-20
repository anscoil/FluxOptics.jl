# Plotting

Visualization tools for optical fields and components.

## Overview

The `Plotting` module provides:
- **Field visualization**: Heatmaps for 2D optical fields
- **Multiple representations**: Intensity, phase, real/imaginary, complex
- **Stack visualization**: Animated sliders for field evolution
- **Component visualization**: Display phase masks, DOEs, and other 2D components
- **Complex colormaps**: HSV-based visualization for complex fields

**Note:** This module is loaded conditionally when `using Makie` is called.

## Examples

### Single Field Visualization

```@example plotting1
using FluxOptics, CairoMakie

# Create Gaussian beam
xv, yv = spatial_vectors(256, 256, 1.0, 1.0)
u = ScalarField(Gaussian(30.0)(xv, yv), (1.0, 1.0), 1.064)

# Visualize intensity
visualize(u, intensity; height=120)
```

```@example plotting1
# Visualize phase (with quadratic phase)
quadratic_phase =  cis.(0.002 .* (xv.^2 .+ yv'.^2))
u_phase = ScalarField(Gaussian(30.0)(xv, yv) .* quadratic_phase, (1.0, 1.0), 1.064)
visualize(u_phase, phase; colormap=:twilight, height=120)
```

### Multiple Representations

```@example plotting2
using FluxOptics, CairoMakie

# Create vortex beam
xv, yv = spatial_vectors(256, 256, 1.0, 1.0)
gaussian = Gaussian(30.0)(xv, yv)
vortex = gaussian .* cis.(atan.(yv', xv))
u_vortex = ScalarField(vortex, (1.0, 1.0), 1.064)

# Show multiple views
visualize(u_vortex, (intensity, phase, real, imag); 
          colormap=(:inferno, :viridis, :RdBu, :RdBu), height=120)
```

### Complex Field Visualization

```@example plotting3
using FluxOptics, CairoMakie

# Create Laguerre-Gaussian mode
xv, yv = spatial_vectors(256, 256, 1.0, 1.0)
lg = LaguerreGaussian(30.0, 2, 2)
u_lg = ScalarField(lg(xv, yv), (1.0, 1.0), 1.064)

# Complex visualization with HSV colormap
visualize(u_lg, (identity, identity); colormap=(:dark, :light), height=120)
```

### Field Comparison

```@example plotting4
using FluxOptics, CairoMakie

# Create initial and propagated fields
xv, yv = spatial_vectors(256, 256, 1.0, 1.0)
u_initial = ScalarField(Gaussian(20.0)(xv, yv), (1.0, 1.0), 1.064)
u_prop = propagate(u_initial, ASProp(u_initial, 1000.0), Forward)

# Compare side by side
visualize((u_initial, u_prop), (intensity, phase);
          colormap=(:inferno, :viridis), height=120)
```

### Component Visualization

```@example plotting5
using FluxOptics, CairoMakie

# Create phase mask
xv, yv = spatial_vectors(256, 256, 2.0, 2.0)
u = ScalarField(ones(ComplexF64, 256, 256), (2.0, 2.0), 1.064)

# Quadratic phase mask
phase_mask = Phase(u, (x, y) -> 0.001*(x^2 + y^2))

# Visualize the component directly
visualize(phase_mask, (identity, x -> phase(cis.(x))); show_colorbars=true, height=120)
```

### Multi-Mode Fields

```@example plotting6
using FluxOptics, CairoMakie

# Create Hermite-Gaussian modes
xv, yv = spatial_vectors(128, 128, 1.0, 1.0)
hg00 = HermiteGaussian(15.0, 0, 0)(xv, yv, Shift2D(-15.0, 0))
hg10 = HermiteGaussian(15.0, 1, 0)(xv, yv)
hg01 = HermiteGaussian(15.0, 0, 1)(xv, yv)
hg11 = HermiteGaussian(15.0, 1, 1)(xv, yv, Shift2D(20.0, 0) ∘ Rot2D(π/4))

# Stack modes
modes = cat(hg00, hg10, hg01, 1.3*hg11; dims=3)
u_modes = ScalarField(modes, (1.0, 1.0), 1.064)

# Total intensity
visualize(u_modes, intensity; colormap=:inferno, height=120)
```

### Propagation Sequence

```@example plotting
using FluxOptics, CairoMakie

# Create propagation sequence
xv, yv = spatial_vectors(256, 256, 1.0, 1.0)
u0 = ScalarField(Gaussian(20.0)(xv, yv), (1.0, 1.0), 1.064)

# Propagate at different distances
distances = [0, 100, 500, 1000, 2000]
propagation = [propagate(u0, ASProp(u0, z), Forward) for z in distances]

# Visualize evolution
visualize(propagation, (intensity, phase); colormap=(:inferno, twilight_shifted), height=120)
```

## Key Functions

- [`visualize`](@ref): Display field(s) with specified representation(s)
- [`visualize_slider`](@ref): Interactive slider for field sequences (**requires GLMakie**)

## Visualization Functions

Common functions for field visualization:
- `intensity`: Total intensity |u|²
- `phase`: Phase angle
- `real`: Real part
- `imag`: Imaginary part
- `abs`: Magnitude
- `identity` / `complex`: Complex field with HSV colormap

## See Also

- [Fields](../fields/index.md) for ScalarField type
- [Makie.jl](https://docs.makie.org/stable/) for plotting backend documentation

## Index

```@index
Modules = [FluxOptics.Plotting]
Order = [:type, :function]
```
