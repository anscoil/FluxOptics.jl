# Modes API

## Gaussian Beam Modes

### Gaussian Beams

```@docs
Gaussian1D
Gaussian
```

#### Examples

```julia
using FluxOptics

# Simple 2D Gaussian at focus
g = Gaussian(25.0)  # 25 μm waist
xv, yv = spatial_vectors(128, 128, 2.0, 2.0)
field = g(xv, yv)

# Elliptical Gaussian
g_ellip = Gaussian(20.0, 30.0)  # Different waists in x and y
field_ellip = g_ellip(xv, yv)

# Propagated Gaussian
g_prop = Gaussian(25.0, 1.064, 1000.0)  # At z=1mm from waist
field_prop = g_prop(xv, yv)
```

### Hermite-Gaussian Modes

```@docs
HermiteGaussian1D
HermiteGaussian
hermite_gaussian_groups
```

#### Examples

```julia
# Single HG mode
hg21 = HermiteGaussian(15.0, 2, 1)  # HG₂₁
field_hg = hg21(xv, yv)

# Complete set of modes up to order N
N = 4
modes = hermite_gaussian_groups(12.0, N)
# Returns HG₀₀, HG₁₀, HG₀₁, HG₂₀, HG₁₁, HG₀₂, ...

# Check orthogonality
mode_fields = [m(xv, yv) for m in modes]
dot(mode_fields[1], mode_fields[2]) * 2.0 * 2.0  # ≈ 0
```

### Laguerre-Gaussian Modes

```@docs
LaguerreGaussian
```

#### Examples

```julia
# Fundamental mode (equivalent to Gaussian)
lg00 = LaguerreGaussian(20.0, 0, 0)
field_lg00 = lg00(xv, yv)

# Vortex beam (default)
lg01 = LaguerreGaussian(20.0, 0, 1)  # Donut beam
field_vortex = lg01(xv, yv)

# Even/odd superpositions
lg_even = LaguerreGaussian(20.0, 0, 2; kind=:even)
lg_odd = LaguerreGaussian(20.0, 0, 2; kind=:odd)

# These are real-valued
@assert all(isreal, lg_even(xv, yv))
@assert all(isreal, lg_odd(xv, yv))
```

## Spatial Layouts

### Layout Types

```@docs
Modes.PointLayout
Modes.GridLayout
Modes.TriangleLayout
Modes.CustomLayout
```

### Mode Stack Generation

```@docs
generate_mode_stack
```

#### Examples

```julia
# Replicate same mode at each position
gaussian = Gaussian(10.0)
layout = Modes.GridLayout(2, 3, 80.0, 80.0)
modes = generate_mode_stack(layout, 256, 256, 1.0, 1.0, gaussian)
# Returns 256×256×6 array

# Different mode at each position
hg_modes = [HermiteGaussian(12.0, m, n) 
            for m in 0:2 for n in 0:2]
layout = Modes.GridLayout(3, 3, 60.0, 60.0)
modes = generate_mode_stack(layout, 128, 128, 1.5, 1.5, hg_modes)
# Returns 128×128×9 array

# Single position, multiple modes
lg_modes = [LaguerreGaussian(15.0, 0, l) for l in 0:5]
modes = generate_mode_stack(128, 128, 2.0, 2.0, lg_modes)
# Returns 128×128×6 array (all centered)

# With coordinate transformations
transform = Shift2D(20.0, -10.0) ∘ Rot2D(π/4)
layout_transformed = Modes.GridLayout(3, 3, 50.0, 50.0, transform)
```

## Speckle Generation

```@docs
generate_speckle
```

### Examples

```julia
# 2D speckle with controlled correlation length
speckle = generate_speckle((256, 256), (1.0, 1.0), 1.064, 0.2)
# Returns 256×256 complex array, normalized to unit power

# 1D speckle
speckle_1d = generate_speckle((512,), (0.5,), 1.064, 0.15)

# 3D speckle
speckle_3d = generate_speckle((64, 64, 64), (2.0, 2.0, 2.0), 1.064, 0.1)

# With Gaussian envelope
envelope = Gaussian(50.0)
speckle_gauss = generate_speckle(
    (256, 256), (1.0, 1.0), 1.064, 0.2;
    envelope=envelope
)

# Controlling speckle statistics via NA
speckle_fine = generate_speckle((256, 256), (1.0, 1.0), 1.064, 0.8)    # High NA
speckle_coarse = generate_speckle((256, 256), (1.0, 1.0), 1.064, 0.05)  # Low NA

# Using with ScalarField
speckle_data = generate_speckle((128, 128), (2.0, 2.0), 1.064, 0.15;
                                envelope=Gaussian(30.0))
u = ScalarField(speckle_data, (2.0, 2.0), 1.064)
```

## Technical Notes

### Mode Normalization
- All modes are normalized to unit power by default
- Power = ∫∫ |u(x,y)|² dx dy = 1
- For custom normalization, use `norm_constant` parameter

### Mode Propagation
- Use `λ` and `z` parameters for Gaussian beam propagation
- Includes Gouy phase shift
- `constant_phase=true` includes exp(ikz) term

### Layout Centering
- Layouts are centered at (0, 0) by default
- Use coordinate transformations to shift layouts

### Speckle Statistics
- Correlation length ≈ λ / NA
- High NA → fine speckle (small correlation length)
- Low NA → coarse speckle (large correlation length)
- Speckle is fully developed (unit contrast) without envelope

## See Also

- [GridUtils](../gridutils/index.md) for coordinate systems
- [Fields](../fields/index.md) for ScalarField operations
