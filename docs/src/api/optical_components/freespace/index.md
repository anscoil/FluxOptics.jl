# Free-Space Propagators

Field propagation methods for homogeneous media.

## Overview

The `Free-Space Propagators` module provides:
- **Angular Spectrum method** for general propagation
- **Rayleigh-Sommerfeld diffraction** for short distances
- **Collins integral (ABCD)** for paraxial systems with magnification
- **Fourier lenses** for ideal optical Fourier transforms
- **Geometric shifts** for comparison with ray optics

## Quick Example

```julia
using FluxOptics

u = ScalarField(ones(ComplexF64, 256, 256), (2.0, 2.0), 1.064)
xv, yv = spatial_vectors(256, 256, 2.0, 2.0)
u.electric .= Gaussian(20.0)(xv, yv)

# Angular Spectrum propagation
prop = ASProp(u, 1000.0)
u_out = propagate(u, prop, Forward)

# Fourier lens
lens = FourierLens(u, (1.0, 1.0), 1000.0)

# System with lens and propagation
system = ScalarSource(u) |> lens |> ASProp(u, 1000.0)
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

**Rayleigh-Sommerfeld (RSProp):** More accurate for short distances but requires dx < λ/2. Use when sampling is very fine.

**Collins/Fourier (CollinsProp, FourierLens):** ABCD systems with grid magnification. Essential for Fourier optics and imaging systems.

**Paraxial (ParaxialProp):** Convenience wrapper that selects ASProp or CollinsProp based on whether magnification is needed.

**Geometric (ShiftProp):** Ray optics approximation, no diffraction. For comparison only.

## See Also

- [Bulk Propagators](../bulk/index.md) for inhomogeneous media
- [Core](../core/index.md) for trainability system

## Index

```@index
Modules = [FluxOptics.OpticalComponents]
Pages = ["freespace.md"]
Order = [:type, :function]
```
