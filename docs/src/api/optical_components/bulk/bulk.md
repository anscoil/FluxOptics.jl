# Bulk Propagators API

```@meta
CurrentModule = FluxOptics.OpticalComponents
```

## Types

```@docs
AS_BPM
Shift_BPM
```

## Usage Examples

### Graded-Index Fiber

```julia
using FluxOptics

u = ScalarField(ones(ComplexF64, 256, 256), (2.0, 2.0), 1.064)
xv, yv = spatial_vectors(256, 256, 2.0, 2.0)

# Parabolic index profile
thickness = 1000.0
n_slices = 100
r = sqrt.(xv.^2 .+ yv'.^2)
dn_fiber = -0.01 * (r/50).^2
dn_3d = repeat(dn_fiber, 1, 1, n_slices)

bpm = AS_BPM(u, thickness, 1.5, dn_3d)
u_out = propagate(u, bpm, Forward)
```

### Waveguide Propagation

```julia
# Step-index waveguide
core_radius = 10.0
dn_waveguide = [sqrt(x^2 + y^2) < core_radius ? 0.02 : 0.0 
                for x in xv, y in yv]
dn_3d = repeat(dn_waveguide, 1, 1, n_slices)

bpm_wg = AS_BPM(u, thickness, 1.5, dn_3d)
```

### Trainable Refractive Index

```julia
# Optimize refractive index profile
dn_init = zeros(256, 256, 100)
bpm_opt = AS_BPM(u, 1000.0, 1.5, dn_init; trainable=true, buffered=true)

system = source |> bpm_opt
# ... optimization loop ...
```

### Diffraction vs Geometric Propagation

```julia
# Tilted beam
u_tilted = ScalarField(gaussian(xv, yv), (2.0, 2.0), 1.064; tilts=(0.01, 0.0))

# With diffraction
bpm_diffraction = AS_BPM(u_tilted, thickness, 1.5, dn_3d)
u_diff = propagate(u_tilted, bpm_diffraction, Forward)

# Geometric only (backprojection)
bpm_geometric = Shift_BPM(u_tilted, thickness, dn_3d)
u_geom = propagate(u_tilted, bpm_geometric, Forward)

# Quantify diffraction contribution
diffraction_effect = intensity(u_diff) - intensity(u_geom)
```

## Technical Notes

### Split-Step Method

AS_BPM uses split-step Fourier method:
1. Apply phase shift from Δn: `exp(i k₀ Δn dz cos(θ))`
2. Propagate dz in background medium (n₀)
3. Repeat for all slices

**Note:** Implementation adds half-steps (dz/2) before the first and after the last phase mask for symmetry, though this has minimal impact on results.

Cosine correction accounts for oblique propagation in tilted beams.

### Slice Discretization

More slices → better accuracy but slower:
- Rule of thumb: `dz < λ / (2 max|Δn|)`
- For weak variations: 10-50 slices often sufficient
- For strong gradients: 100-1000 slices may be needed

### Paraxial vs Non-Paraxial

- `paraxial=false` (default): Full Angular Spectrum, handles large angles
- `paraxial=true`: Paraxial approximation, faster but limited to small angles

### Geometric Shift (Shift_BPM)

Pure ray optics, no diffraction:
- Uses tilt metadata from ScalarField
- Ignores phase gradients in complex field
- Equivalent to tomographic backprojection
- Generally inferior to AS_BPM in optical regimes

**Why people still use backprojection:** See discussion below.

## Backprojection vs Diffraction

Backprojection (Shift_BPM) is still used in some fields despite ignoring diffraction:

**Valid in X-ray CT:**
- Very short wavelengths (λ ≈ 0.1 nm)
- Diffraction effects negligible
- Geometric optics sufficient

**Why it persists elsewhere:**
- Historical inertia from medical imaging
- Established software ecosystems
- Algorithm maturity and validation

**For optical wavelengths** (visible, NIR, fiber optics):
- Diffraction is significant and cannot be ignored
- AS_BPM provides physically correct results
- Shift_BPM useful only for comparison/validation

## See Also

- [ASProp](@ref) for homogeneous propagation
- [Core](../core/index.md) for trainability
