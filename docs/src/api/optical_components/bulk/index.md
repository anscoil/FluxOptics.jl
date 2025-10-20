# Bulk Propagators

Beam Propagation Method for inhomogeneous media.

## Overview

The `Bulk Propagators` module provides:
- **Beam Propagation Method** with Angular Spectrum
- **Split-step propagation** through varying refractive index
- **Geometric shift propagation** for comparison
- **Trainable refractive index** for inverse design

## Quick Example

```@example BPM
using FluxOptics, CairoMakie

xv, yv = spatial_vectors(256, 256, 1.0, 1.0)
u = ScalarField(LaguerreGaussian(25.0, 2, 1)(xv, yv), (1.0, 1.0), 1.064)

# Graded-index medium
thickness = 1500.0  # μm
n_slices = 100
r = sqrt.(xv.^2 .+ yv'.^2)
dn = -0.008 * (r/50).^2  # Parabolic profile
dn_3d = repeat(dn, 1, 1, n_slices)

n0 = 1.5  # Bulk refractive index
bpm = AS_BPM(u, thickness, n0, dn_3d)
u_out = propagate(u, bpm, Forward)

visualize((u, u_out), (intensity, phase); colormap=(:inferno, :viridis), height=120)
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
