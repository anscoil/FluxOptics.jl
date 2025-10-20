# Free-Space Propagators

Field propagation methods for homogeneous media.

## Overview

The `Free-Space Propagators` module provides:
- **Angular Spectrum method** for general propagation
- **Rayleigh-Sommerfeld diffraction** for short distances
- **Collins integral (ABCD)** for paraxial systems with magnification
- **Fourier lenses** for ideal optical Fourier transforms
- **Geometric shifts** without diffraction effects

## Examples

### Angular Spectrum

```@example AS
using FluxOptics, CairoMakie

# Create Gaussian beam
xv, yv = spatial_vectors(256, 256, 2.0, 2.0)
u = ScalarField(Gaussian(75.0, 50.0)(xv, yv), (2.0, 2.0), 1.064)

# Basic propagation
prop = ASProp(u, 12000.0)
u_out = propagate(u, prop, Forward)

# Compare beams before and after propagation
visualize((u, u_out), (intensity, phase); colormap=(:inferno, :viridis), height=120)
```

```@example AS
# Propagation in different medium
prop_glass = ASProp(u, 12000.0; n0=1.5)
u_glass = propagate(u, prop_glass, Forward)

# With spatial filter
filter_lp = (fx, fy) -> sqrt(fx^2 + fy^2) < 0.008 ? 1.0 : 0.0
prop_filtered = ASProp(u, 12000.0; filter=filter_lp)
u_filtered = propagate(u, prop_filtered, Forward)

# Compare peak intensities
visualize((u_glass, u_filtered), (intensity, phase);
	colormap=(:inferno, :viridis), height=120)
```

### Rayleigh-Sommerfeld

```@example RS
using FluxOptics, CairoMakie

# Create Gaussian beam
xv, yv = spatial_vectors(256, 256, 2.0, 2.0)
u = ScalarField(Gaussian(75.0, 50.0)(xv, yv), (2.0, 2.0), 1.064)

as_prop = ASProp(u, 50000.0)
rs_prop = RSProp(u, 50000.0)

u_as = propagate(u, as_prop, Forward)
u_rs = propagate(u, rs_prop, Forward)

# Compare beams propagated with AS and RS
visualize((u_as, u_rs), (intensity, phase); colormap=(:inferno, :viridis), height=120)
```

### ABCD Systems and Fourier Lenses

```@example ABCD
using FluxOptics, CairoMakie

xv, yv = spatial_vectors(256, 256, 1.0, 1.0)
u = ScalarField(Gaussian(20.0)(xv, yv), (1.0, 1.0), 1.064)

# Telescope comparison: Two approaches
f1, f2 = 100.0, 200.0  # Magnification = f2/f1 = 2

# Approach 1: Using FourierLens components
source = ScalarSource(u)
lens1 = FourierLens(u, (0.1, 0.1), f1)  # Magnification of the intermediate grid
lens2 = FourierLens(u, (0.1, 0.1), (1.5, 1.5), f2)  # Magnification of the output grid
telescope_fourier = source |> lens1 |> lens2

# Approach 2: Using ABCD matrix formulation
# Define matrices
M_prop1 = [1 f1; 0 1]           # Propagation to lens1
M_lens1 = [1 0; -1/f1 1]        # First lens
d = f1 + f2                     # Telescope spacing
M_prop12 = [1 d; 0 1]           # Propagation from lens1 to lens2
M_lens2 = [1 0; -1/f2 1]        # Second lens
M_prop2 = [1 5000; 0 1]         # Auxiliary last propagation to avoid B = 0

# Composite ABCD matrix for telescope
M = M_prop2 * M_lens2 * M_prop12 * M_lens1 * M_prop1

# Use CollinsProp
collins = CollinsProp(u, (1.5, 1.5), (M[1,1], M[1,2], M[2,2]))
telescope_abcd = source |> collins |>
	ASProp(u, (1.5, 1.5), f2-5000)  # Correct last propagation

# Execute both systems
u_out1 = telescope_fourier().out
u_out2 = telescope_abcd().out

visualize((u, -u_out1, -im*u_out2), (intensity, phase);
    colormap=(:inferno, :viridis), height=120)
```

```@example ABCD
coupling_efficiency(u_out1, u_out2)
```

## Key Types

- [`ASProp`](@ref): Angular Spectrum propagation (static)
- [`ASPropZ`](@ref): Angular Spectrum with trainable distance
- [`RSProp`](@ref): Rayleigh-Sommerfeld diffraction
- [`CollinsProp`](@ref): ABCD matrix propagation with resampling
- [`FourierLens`](@ref): Ideal Fourier lens
- [`ParaxialProp`](@ref): Paraxial propagation (convenience wrapper)
- [`ShiftProp`](@ref): Geometric shift (no diffraction)

## Method Selection

**Angular Spectrum (ASProp):** General-purpose, handles both paraxial and non-paraxial regimes. Default choice for most applications.

**Rayleigh-Sommerfeld (RSProp):** Prevents aliasing for large propagation distances but requires finer sampling for short distances.

**Collins Integral (CollinsProp, FourierLens):** ABCD systems with grid magnification. Essential for Fourier optics and imaging systems. Works only when B ≠ 0.

**Paraxial (ParaxialProp):** Convenience wrapper that selects ASProp or CollinsProp based on whether magnification is needed.

**Geometric (ShiftProp):** Pure geometric shift, no diffraction. Useful for tomography phase projection comparisons.

## See Also

- [Bulk Propagators](../bulk/index.md) for inhomogeneous media
- [Core](../core/index.md) for trainability system

## Index

```@index
Modules = [FluxOptics.OpticalComponents]
Pages = ["freespace.md"]
Order = [:type, :function]
```
