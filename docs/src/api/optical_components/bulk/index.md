# Bulk Propagators

Beam Propagation Method for inhomogeneous media.

## Overview

The `Bulk Propagators` module provides:
- **Beam Propagation Method** with Angular Spectrum
- **Split-step propagation** through varying refractive index
- **Geometric shift propagation** for comparison
- **Trainable refractive index** for inverse design

## Quick Examples

### Graded-Index Fiber Propagation
Propagate a Laguerre-Gaussian beam through a parabolic graded-index fiber over 1.5 mm.

```@example BPM
using FluxOptics, CairoMakie

ns = (256, 256)
ds = (1.0, 1.0)
xv, yv = spatial_vectors(ns, ds)
u = ScalarField(LaguerreGaussian(25.0, 2, 1)(xv, yv), ds, 1.064)

# Graded-index medium
thickness = 1500.0  # μm
n_slices = 100
r = sqrt.(xv.^2 .+ yv'.^2)
dn = -0.008 * (r/50).^2  # Parabolic profile
dn_3d = repeat(dn, 1, 1, n_slices)

n0 = 1.5  # Bulk refractive index
bpm = AS_BPM(u, thickness, n0, dn_3d)
u_out = propagate(u, bpm)

visualize((u, u_out), (intensity, phase); colormap=(:inferno, :viridis), height=120)
```

### Freeform Interface Visualization
Visualize the smoothstep partition of a spherical glass/air interface, showing how
the transition is distributed across the longitudinal slices.

```@example FS_WPM
using FluxOptics, CairoMakie

ns = (128, 128)
ds = (0.5, 0.5)

u = ScalarField(ones(ComplexF32, ns), ds, 1.55)

# Spherical freeform surface
thickness = 50.0
dz = 0.5
n1 = 1.5
n2 = 1.0
Rc = 100.0
f = (x, y) -> sqrt(max(0.0, Rc^2 - x^2 - y^2)) - Rc
p = FS_WPM(u, thickness, dz, n1, n2, f; nz=4)

# Visualize the partition volume — central cross-section
V = smoothstep_partition(p)
Vd = smoothstep_partition(p; derivative=true)
xv, = spatial_vectors(ns[1], ds[1])
zv = p.dz .* (0:p.n_slices)

fig = Figure(size=(480, 220))
titles = ["Partition", "Derivative"]
cmaps = [:RdBu, :inferno]
cranges = [(0, 1), nothing]
datas = [V[:, 64, :]', Vd[:, 64, :]']

for (k, (data, cmap, crange, title)) in enumerate(zip(datas, cmaps, cranges, titles))
    cell = fig[1, k] = GridLayout()
    ax, hm = heatmap(cell[1, 1], zv, xv, data;
                     colormap=cmap,
                     colorrange=isnothing(crange) ? Makie.automatic : crange)
    ax.aspect = DataAspect()
    ax.xlabel = "z (μm)"
    ax.ylabel = k == 1 ? "x (μm)" : ""
    ax.title = title
    Colorbar(cell[1, 2], hm; width=10)
end
fig
```

## Key Types

- [`AS_BPM`](@ref): BPM with Angular Spectrum propagation
- [`Shift_BPM`](@ref): BPM with geometric shifts (no diffraction)
- [`FS_WPM`](@ref): Freeform surface wave propagation beyond the paraxial approximation

## Key Functions

- [`smoothstep_partition`](@ref): Visualize the partition function volume of a `FS_WPM` component

## See Also

- [Free-Space Propagators](../freespace/index.md) for homogeneous propagation
- [Core](../core/index.md) for trainability system

## Index

```@index
Modules = [FluxOptics.OpticalComponents]
Pages = ["bulk.md"]
Order = [:type, :function]
```
