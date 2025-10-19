# Modes

Optical mode generation and spatial layouts.

## Overview

The `Modes` module provides:
- **Gaussian beam families** (Gaussian, Hermite-Gaussian, Laguerre-Gaussian)
- **Spatial layouts** for mode composition and multi-mode configurations
- **Speckle generation** with controlled statistics and envelope
- **Mode stacks** for multi-mode generation

## Examples

### Gaussian Beams

```@example modes
using FluxOptics
using CairoMakie

# Setup coordinates
xv, yv = spatial_vectors(256, 256, 1.0, 1.0)

# Gaussian propagated from focus (spherical phase front)
g = Gaussian(25.0, 1.064, 1000.0)  # 1000 µm from waist
field_gauss = g(xv, yv)

# Elliptical Gaussian (cylindrical phase fronts)
g_ellip = Gaussian(20.0, 30.0, 1.064, 1000.0)
field_ellip = g_ellip(xv, yv)

# Visualize
visualize((field_gauss, field_ellip), (intensity, phase); 
    colormap=(:inferno, :viridis),
    height=120)
```

### Hermite-Gaussian Gallery

```@example modes
# Generate HG modes of different orders
hg_modes = [
    (HermiteGaussian(20.0, 0, 0), "HG₀₀"),
    (HermiteGaussian(20.0, 1, 0), "HG₁₀"),
    (HermiteGaussian(20.0, 0, 1), "HG₀₁"),
    (HermiteGaussian(20.0, 2, 0), "HG₂₀"),
    (HermiteGaussian(20.0, 1, 1), "HG₁₁"),
    (HermiteGaussian(20.0, 0, 2), "HG₀₂")
]

# Create field stack
hg_fields = [mode(xv, yv) for (mode, _) in hg_modes]

# Visualize intensity patterns
fig = Figure(size=(1200, 800))
for (i, (field, (_, label))) in enumerate(zip(hg_fields, hg_modes))
    row = (i - 1) ÷ 3 + 1
    col = (i - 1) % 3 + 1
    ax = Axis(fig[row, col], title=label, titlesize=30, aspect=DataAspect())
    heatmap!(ax, xv, yv, abs2.(field); colormap=:inferno)
    hidedecorations!(ax)
end
fig
```

### Laguerre-Gaussian Gallery

```@example modes
# Generate LG modes with vortex phases
lg_modes = [
    (LaguerreGaussian(20.0, 0, 0), "LG₀₀"),
    (LaguerreGaussian(20.0, 0, 1), "LG₀₁"),
    (LaguerreGaussian(20.0, 0, 2), "LG₀₂"),
    (LaguerreGaussian(20.0, 1, 0), "LG₁₀"),
    (LaguerreGaussian(20.0, 1, 1), "LG₁₁"),
    (LaguerreGaussian(20.0, 0, -2), "LG₀₋₂")
]

# Create fields
lg_fields = [mode(xv, yv) for (mode, _) in lg_modes]

# Visualize intensity and phase
fig = Figure(size=(1200, 1600))
for (i, (field, (_, label))) in enumerate(zip(lg_fields, lg_modes))
    row = (i - 1) ÷ 3 + 1
    col = (i - 1) % 3 + 1
    
    # Intensity
    ax_int = Axis(fig[2*row-1, col], title=label, titlesize=30, aspect=DataAspect())
    heatmap!(ax_int, xv, yv, abs2.(field); colormap=:inferno)
    hidedecorations!(ax_int)
    
    # Phase
    ax_phase = Axis(fig[2*row, col], aspect=DataAspect())
    heatmap!(ax_phase, xv, yv, phase(field); colormap=:viridis)
    hidedecorations!(ax_phase)
end

# Add row labels
Label(fig[1, 0], "Intensity", rotation=π/2, fontsize=30, tellheight=false)
Label(fig[2, 0], "Phase", rotation=π/2, fontsize=30, tellheight=false)
Label(fig[3, 0], "Intensity", rotation=π/2, fontsize=30, tellheight=false)
Label(fig[4, 0], "Phase", rotation=π/2, fontsize=30, tellheight=false)

fig
```

### Spatial Layouts

```@example modes
# 3×3 grid of Gaussian modes
gaussian = Gaussian(10.0)
layout = Modes.GridLayout(3, 3, 60.0, 60.0)

# Generate mode stack
xv_layout, yv_layout = spatial_vectors(256, 256, 1.0, 1.0)
mode_stack = generate_mode_stack(layout, 256, 256, 1.0, 1.0, gaussian)

# Visualize first mode and total intensity
field_first = mode_stack[:, :, 1]
field_total = sum(mode_stack, dims=3)[:, :, 1]

visualize(((field_first, field_total),), intensity; colormap=:inferno, height=120)
```

### Mode Composition

```@example modes
# Different modes at each grid position
hg_mode_list = [HermiteGaussian(12.0, m, n) for m in 0:2 for n in 0:2]
layout_hg = Modes.GridLayout(3, 3, 60.0, 60.0)

mode_stack_hg = generate_mode_stack(layout_hg, 256, 256, 1.0, 1.0, hg_mode_list)

# Normalize by peak intensity for better visualization
for i in 1:size(mode_stack_hg, 3)
    mode_stack_hg[:, :, i] ./= maximum(abs, mode_stack_hg[:, :, i])
end

# Visualize coherent superposition
field_superposition = sum(mode_stack_hg, dims=3)[:, :, 1]

visualize(field_superposition, (intensity, phase); colormap=(:inferno, :viridis), height=120)
```

### Speckle Patterns

```@example modes
# Generate speckle with different correlation lengths
speckle_fine = generate_speckle((256, 256), (1.0, 1.0), 1.064, 0.25)  # Medium-fine
speckle_coarse = generate_speckle((256, 256), (1.0, 1.0), 1.064, 0.05)  # Low NA

# With Gaussian envelope
envelope = Gaussian(80.0)
speckle_envelope = generate_speckle((256, 256), (1.0, 1.0), 1.064, 0.2; envelope=envelope)

# Visualize
visualize((speckle_fine, speckle_coarse, speckle_envelope), (intensity, complex);
	colormap=(:inferno, :dark), height=120)
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
