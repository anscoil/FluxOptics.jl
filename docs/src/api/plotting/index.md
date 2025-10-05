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

## Quick Example

```julia
using FluxOptics
using GLMakie  # or CairoMakie, WGLMakie

# Visualize a single field
u = ScalarField(gaussian_data, (2.0, 2.0), 1.064)
visualize(u, intensity)

# Multiple representations
visualize(u, (intensity, phase, real, imag))

# Visualize propagation sequence
propagation_sequence = [u0, u1, u2, u3, u4]
visualize_slider(propagation_sequence, intensity)
```

## Key Functions

- [`visualize`](@ref): Display field(s) with specified representation(s) (CairoMakie or GLMakie)
- [`visualize_slider`](@ref): Interactive slider for field sequences (**requires GLMakie**)

## When to Use What

**Use `visualize` when:**
- Displaying single fields or small sets
- Creating static figures for papers/reports (CairoMakie)
- Comparing a few fields side-by-side

**Use `visualize_slider` when:**
- Viewing large field sequences (propagation, optimization iterations)
- Need interactive navigation through many frames
- Avoid displaying all frames simultaneously

## Visualization Functions

Common functions for field visualization:
- `intensity`: Total intensity |u|²
- `phase`: Phase angle
- `real`: Real part
- `imag`: Imaginary part
- `abs`: Magnitude
- `identity`: Complex field with HSV colormap

## See Also

- [Fields](../fields/index.md) for ScalarField type
- [Makie.jl](https://docs.makie.org/stable/) for plotting backend documentation

## Index

```@index
Modules = [FluxOptics.Plotting]
Order = [:type, :function]
```
