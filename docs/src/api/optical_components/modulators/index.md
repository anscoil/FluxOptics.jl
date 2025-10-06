# Modulators

Phase and amplitude modulation components.

## Overview

The `Modulators` module provides:
- **Phase masks** for wavefront shaping
- **Amplitude masks** for apertures and apodization
- **Diffractive optical elements** with thin element approximation
- **Reflective elements** for mirror-based systems

## Quick Example

```julia
using FluxOptics

xv, yv = spatial_vectors(256, 256, 2.0, 2.0)
u = ScalarField(ones(ComplexF64, 256, 256), (2.0, 2.0), 1.064)

# Phase mask (lens-like)
phase = Phase(u, (x, y) -> π/(1000^2) * (x^2 + y^2))

# Circular aperture
aperture = Mask(u, (x, y) -> sqrt(x^2 + y^2) < 50.0 ? 1.0 : 0.0)

# Diffractive element
grating = TeaDOE(u, 0.5, (x, y) -> 0.5 * sin(2π * x / 50))

# Use in system
system = ScalarSource(u) |> phase |> aperture |> ASProp(u, 1000.0)
```

## Key Types

- [`Phase`](@ref): Pure phase modulation
- [`Mask`](@ref): Amplitude or complex transmission
- [`TeaDOE`](@ref): Diffractive optical element
- [`TeaReflector`](@ref): Reflective element

## See Also

- [Core](../core/index.md) for trainability and component interface
- [Fourier](../fourier/index.md) for frequency-domain modulation
- [Sources](../sources/index.md) for field generation

## Index

```@index
Modules = [FluxOptics.OpticalComponents]
Pages = ["modulators.md"]
Order = [:type, :function]
```
