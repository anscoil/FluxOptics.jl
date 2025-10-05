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

## Examples

### Single Field Visualization

```julia
using FluxOptics, GLMakie

# Create field
u = ScalarField(ones(ComplexF64, 256, 256), (2.0, 2.0), 1.064)
xv, yv = spatial_vectors(256, 256, 2.0, 2.0)
u.electric .= Gaussian(30.0)(xv, yv)

# Visualize intensity
fig = visualize(u, intensity)
display(fig)

# Visualize phase
fig = visualize(u, phase)

# Complex field with HSV colormap
fig = visualize(u, identity)  # Hue=phase, Value=magnitude
```

### Multiple Representations

```julia
# Show multiple views side-by-side
fig = visualize(u, (intensity, phase, real, imag))
display(fig)

# Different colormaps per view
fig = visualize(u, (intensity, phase); 
                colormap=(:viridis, :twilight))
```

### Multi-Mode Fields

```julia
# Visualize all modes
data = generate_mode_stack(layout, 128, 128, 2.0, 2.0, Gaussian(15.0))
u_multi = ScalarField(data, (2.0, 2.0), 1.064)

# Each mode shown separately
fig = visualize(u_multi, intensity)
```

### Field Sequences with Slider

```julia
# Propagation sequence
distances = range(0, 2000, length=50)
u0 = ScalarField(gaussian_data, (2.0, 2.0), 1.064)

propagation = [propagate(u0, ASProp(u0, z), Forward) for z in distances]

# Interactive slider
fig = visualize_slider(propagation, intensity)
display(fig)

# Multiple representations with slider
fig = visualize_slider(propagation, (intensity, phase))
```

### Component Visualization

```julia
# Visualize phase mask
phase_mask = Phase(u, (x, y) -> 0.01*(x^2 + y^2); trainable=true)
fig = visualize(phase_mask, identity)

# Visualize TeaDOE surface
doe = TeaDOE(u, 0.5, (x, y) -> sin(2π*x/50)*sin(2π*y/50); trainable=true)
fig = visualize(doe, real)
```

### Comparison Visualization

```julia
# Compare target vs optimized
target = ScalarField(target_data, (2.0, 2.0), 1.064)
optimized = system()

# Stack vertically for comparison
fig = visualize([target, optimized], (intensity, phase))
display(fig)
```

### Custom Transformations

```julia
# Custom visualization function
log_intensity(u) = log10.(intensity(u) .+ 1e-10)

fig = visualize(u, log_intensity)

# Multiple custom functions
power_spectrum(u) = abs2.(fftshift(fft(u.electric)))
fig = visualize(u, (intensity, power_spectrum))
```

### Styling Options

```julia
# Larger figures
fig = visualize(u, intensity; height=400)

# Show colorbars
fig = visualize(u, (intensity, phase); 
                show_colorbars=true)

# Custom colormap
fig = visualize(u, intensity; colormap=:inferno)
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

- `visualize`: Displays all fields simultaneously - can be slow/memory-intensive for large stacks
- `visualize_slider`: Only renders current frame - efficient for large sequences
- Arrays are collected (GPU → CPU) automatically if needed
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

## Advanced Usage

### Custom Multi-Panel Layouts

```julia
# Different functions for each mode
mode_funcs = (intensity, phase, real)
fig = visualize(u_multi, mode_funcs)
```

### Saving Figures

```julia
using CairoMakie  # For saving

fig = visualize(u, intensity)
save("field_intensity.png", fig)
save("field_intensity.pdf", fig)  # Vector format
```

### Animation

```julia
using GLMakie

# Create propagation sequence
sequence = [propagate(u0, ASProp(u0, z), Forward) 
            for z in range(0, 2000, length=100)]

# Manual animation
fig = Figure()
ax = Axis(fig[1, 1])
hidedecorations!(ax)

for (i, u) in enumerate(sequence)
    empty!(ax)
    heatmap!(ax, intensity(u))
    sleep(0.05)  # Frame delay
end
```

## See Also

- [Fields](../fields/index.md) for ScalarField operations
- [Makie.jl documentation](https://docs.makie.org/stable/) for advanced plotting
- [ColorSchemes.jl](https://juliagraphics.github.io/ColorSchemes.jl/stable/) for available colormaps
