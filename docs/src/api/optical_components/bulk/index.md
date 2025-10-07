# Bulk Propagators

Beam Propagation Method for inhomogeneous media.

## Overview

The `Bulk Propagators` module provides:
- **Beam Propagation Method** with Angular Spectrum
- **Split-step propagation** through varying refractive index
- **Geometric shift propagation** for comparison
- **Trainable refractive index** for inverse design

## Quick Example

```julia
using FluxOptics

u = ScalarField(ones(ComplexF64, 256, 256), (2.0, 2.0), 1.064)

# Graded-index medium
thickness = 1000.0  # μm
n_slices = 100
xv, yv = spatial_vectors(256, 256, 2.0, 2.0)
r = sqrt.(xv.^2 .+ yv'.^2)
dn = -0.01 * (r/50).^2  # Parabolic profile
dn_3d = repeat(dn, 1, 1, n_slices)

bpm = AS_BPM(u, thickness, 1.5, dn_3d)
u_out = propagate(u, bpm, Forward)
```

## Key Types

- [`AS_BPM`](@ref): BPM with Angular Spectrum propagation
- [`Shift_BPM`](@ref): BPM with geometric shifts (no diffraction)

## See Also

- [Free-Space Propagators](../freespace/index.md) for homogeneous propagation
- [Core](../core/index.md) for trainability system

## Index

```@index
Modules = [FluxOptics.OpticalComponents]
Pages = ["bulk.md"]
Order = [:type, :function]
```
