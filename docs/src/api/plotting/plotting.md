# Plotting API

```@meta
CurrentModule = FluxOptics.Plotting
```

**Loading:** This module is loaded conditionally. Use `using Makie` (or `GLMakie`, `CairoMakie`, `WGLMakie`) before using plotting functions.

## Visualization Functions

```@docs
visualize
visualize_slider
```

## Complex Field Colormaps

For complex-valued visualizations (when using `identity` or when function returns complex):

```julia
# Dark colormap (default): Hue=phase, Value=magnitude
fig = visualize(u, identity; colormap=:dark)

# Light colormap: Hue=phase, Saturation=magnitude
fig = visualize(u, identity; colormap=:light)
```

HSV mapping:
- **Hue**: Phase angle (0 to 2π → 0° to 360°)
- **Saturation** (light mode): Magnitude
- **Value** (dark mode): Magnitude

## Technical Notes

### Plottable Types

The following can be visualized:
- `ScalarField`: Optical fields
- `AbstractOpticalComponent`: Components with 2D data (Phase, Mask, TeaDOE)
- `AbstractArray{<:Number, 2}`: Raw 2D arrays

### Visualization Functions

Functions passed to `visualize` must:
- Accept a plottable object as input
- Return a 2D array (real, complex, or RGB)

Common built-in functions:
```julia
intensity(u)  # |u|² - total intensity
phase(u)      # arg(u) - phase in radians
real(u)       # Re(u)
imag(u)       # Im(u)
abs(u)        # |u| - magnitude
identity(u)   # u itself (complex → HSV colormap)
```

### Colormap Selection

- Real-valued outputs: Use standard Makie colormaps (`:viridis`, `:inferno`, `:twilight`, etc.)
- Complex-valued outputs: Use `:dark` or `:light` for HSV mapping
- Per-view colormaps: Pass tuple matching number of views

Valid colormaps from ColorSchemes.jl are automatically checked.

### Layout and Sizing

- `height`: Control figure height (default: 200 for `visualize`, 400 for `visualize_slider`)
- Width automatically computed to maintain aspect ratio
- Multiple views arranged in grid automatically

### Performance

- Arrays are collected (GPU → CPU) automatically
- Consider downsampling very large fields for interactive visualization

### Slider Interaction

`visualize_slider` provides:
- Horizontal slider to navigate through sequence
- Real-time update of all displayed views
- **Requires GLMakie** - interactive backend with slider support
- Useful for propagation sequences, optimization iterations, wavelength sweeps

## Conditional Loading

The Plotting module uses `@require` for conditional loading:

```julia
# In your script/notebook
using FluxOptics
using GLMakie  # Triggers Plotting module load

# Now plotting functions are available
visualize(field, intensity)
```

Supported Makie backends:
- `GLMakie`: Interactive, OpenGL-based (required for `visualize_slider`)
- `CairoMakie`: High-quality static plots (PDF/SVG export) - works with `visualize` only
- `WGLMakie`: Web-based interactive plots (for Pluto.jl, etc.) - limited slider support

## See Also

- [Fields](../fields/index.md) for ScalarField operations
- [Makie.jl documentation](https://docs.makie.org/stable/) for advanced plotting
- [ColorSchemes.jl](https://juliagraphics.github.io/ColorSchemes.jl/stable/) for available colormaps
