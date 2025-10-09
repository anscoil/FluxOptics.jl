# Free-Space Propagators API

```@meta
CurrentModule = FluxOptics.OpticalComponents
```

## Types

```@docs
ASProp
ASPropZ
RSProp
CollinsProp
FourierLens
ParaxialProp
ShiftProp
```

## Usage Examples

### Angular Spectrum

```julia
using FluxOptics

u = ScalarField(ones(ComplexF64, 256, 256), (2.0, 2.0), 1.064)
xv, yv = spatial_vectors(256, 256, 2.0, 2.0)
u.electric .= Gaussian(20.0)(xv, yv)

# Basic propagation
prop = ASProp(u, 1000.0)
u_out = propagate(u, prop, Forward)

# In different medium
prop_glass = ASProp(u, 1000.0; n0=1.5)

# With spatial filter
filter_lp = (fx, fy) -> sqrt(fx^2 + fy^2) < 0.3 ? 1.0 : 0.0
prop_filtered = ASProp(u, 1000.0; filter=filter_lp)
```

### Trainable Distance

```julia
# Optimize propagation distance
prop_z = ASPropZ(u, 500.0; trainable=true)

system = source |> phase |> prop_z
target = ScalarField(target_data, (2.0, 2.0), 1.064)
metric = PowerCoupling(target)

# Distance is optimized via gradients
loss() = 1.0 - metric(system())[]
```

### Rayleigh-Sommerfeld

```julia
# Fine sampling required
u_fine = ScalarField(ones(ComplexF64, 512, 512), (0.5, 0.5), 1.064)
xv, yv = spatial_vectors(512, 512, 0.5, 0.5)
u_fine.electric .= Gaussian(10.0)(xv, yv)

# Short distance propagation
prop_rs = RSProp(u_fine, 50.0)
u_out = propagate(u_fine, prop_rs, Forward)
```

### ABCD Systems

```julia
# Free-space with magnification
collins = CollinsProp(u, (1.0, 1.0), (1, 500.0, 1))

# Imaging system (4f configuration)
f = 1000.0
lens1 = FourierLens(u, (2.0, 2.0), f)
lens2 = FourierLens(u, (2.0, 2.0), f)
system = source |> lens1 |> ASProp(u, 2*f) |> lens2
```

### Fourier Optics

```julia
# Fourier plane with magnification
lens = FourierLens(u, (1.0, 1.0), 1000.0)

# Filter in Fourier plane
fourier_filter = FourierMask(u, (fx, fy) -> sqrt(fx^2 + fy^2) < 0.2 ? 1.0 : 0.0)

# Complete system
system = source |> lens |> fourier_filter |> 
         FourierLens(set_field_data(u, similar(u.electric)), (2.0, 2.0), 1000.0)
```

### Tilted Beams

```julia
# Tilted beam propagation (automatic handling)
u_tilt = ScalarField(gaussian(xv, yv), (2.0, 2.0), 1.064; tilts=(0.01, 0.005))

# ASProp automatically handles tilts
prop = ASProp(u_tilt, 1000.0)
u_out = propagate(u_tilt, prop, Forward)

# Compare with geometric shift
shift = ShiftProp(u_tilt, 1000.0)
u_geom = propagate(u_tilt, shift, Forward)
```

## Technical Notes

### Angular Spectrum Method

Kernel in frequency domain:
- On-axis: `H = exp(i 2π z √(1/λ² - fx² - fy²))`
- Tilted: Modified with tilt offsets

Handles both paraxial and non-paraxial regimes automatically.

### Rayleigh-Sommerfeld

Direct convolution with Green's function:
- `G(r) = (z/r²)(1/r - ik) exp(ikr) / (2π)`

Requires fine sampling (dx < λ/2) to avoid aliasing.

### Collins Integral

ABCD propagation with chirp multiplication and convolution:
1. Input chirp: `exp(i π A/B (x² + y²) / λ)`
2. Convolution with kernel
3. Output chirp: `exp(i π D/B (x'² + y'²) / λ)`
4. Grid resampling

### Grid Resampling

`FourierLens` and `CollinsProp` change grid sampling:
- Input grid: `ds`
- Output grid: `ds'`
- Magnification: `M = ds' / ds`

Grid center (x=0, y=0) is reference, consistent with beam definitions.

### Geometric Shift

Pure translation based on tilt metadata:
- No diffraction
- Shift: `Δx = z tan(θx)`
- Applied as Fourier phase: `exp(-i 2π z tan(θ) fx)`

## See Also

- [Bulk Propagators](../bulk/index.md) for inhomogeneous media
- [Core](../core/index.md) for component interface
