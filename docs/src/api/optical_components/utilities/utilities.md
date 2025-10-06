# Utilities API

```@meta
CurrentModule = FluxOptics.OpticalComponents
```

## Types

```@docs
PadCropOperator
TiltAnchor
BasisProjectionWrapper
```

## Functions

```@docs
pad
crop
make_spatial_basis
make_fourier_basis
```

## Usage Examples

### Pad and Crop

```julia
using FluxOptics

u = ScalarField(ones(ComplexF64, 128, 128), (2.0, 2.0), 1.064)

# Basic padding
u_pad = pad(u.electric, (256, 256))

# Centered padding
offset = ((256-128)÷2, (256-128)÷2)
u_centered = pad(u.electric, (256, 256); offset=offset)

# Cropping
u_crop = crop(u_pad, (128, 128); offset=offset)
```

### PadCropOperator

```julia
# Efficient pad/crop in systems
u_tmp = zeros(ComplexF64, 256, 256)
pad_op = PadCropOperator(u, u_tmp; store_ref=true)
crop_op = adjoint(pad_op)

# Avoid aliasing during propagation
prop = ASProp(set_field_data(u, u_tmp), 1000.0)
system = source |> pad_op |> prop |> crop_op
```

### Tilt Anchors

```julia
# Track tilts in off-axis systems
anchor1 = TiltAnchor(u)
anchor2 = TiltAnchor(u)

system = source |> anchor1 |> 
         propagator1 |> 
         anchor2 |> 
         propagator2
```

### Basis Projection

```julia
# Polynomial basis
n_basis = 10
basis = make_spatial_basis((x, y, n) -> (x^2 + y^2)^n, 
                            (128, 128), (2.0, 2.0), 0:n_basis-1)

# Wrap trainable phase
phase = Phase(u, (x, y) -> 0.0; trainable=true)
wrapper = BasisProjectionWrapper(phase, basis, zeros(n_basis))

# Optimize n_basis coefficients instead of 128×128 pixels
system = source |> wrapper |> propagator
```

## Technical Notes

### PadCropOperator

**Adjoint behavior:**
- `adjoint(pad_op)` swaps pad and crop
- Enables symmetric operations: `source |> pad |> processing |> crop`

**Store reference mode:**
- `store_ref=true`: Reuses original array during crop (zero allocation)
- Only valid for symmetric operations
- Requires pad and crop to use same offset/sizes

**Direction handling:**
- Forward: applies operation based on `ispad` flag
- Backward: swaps operation

### TiltAnchor

Maintains tilt reference through propagation:
- Stores reference tilt at anchor point
- Later components propagate relative to anchor
- Useful in multi-stage off-axis systems

### BasisProjectionWrapper

**Use cases:**
- Inverse problems with ill-posed pixel-wise optimization
- Enforcing smoothness constraints
- Reducing parameter count for faster convergence

**Basis selection:**
- Polynomials: smooth, global features
- Zernike: optical aberrations
- Fourier: periodic structures
- Custom: problem-specific functions

**Performance:**
- Optimization on N_basis coefficients vs N_pixels
- Typical: N_basis ≪ N_pixels (10-100 vs 10⁴-10⁶)
- Faster convergence, better generalization

## See Also

- [Fields](../../fields/index.md) for ScalarField operations
- [Modulators](../modulators/index.md) for components to wrap
- [System](../system/index.md) for composition
