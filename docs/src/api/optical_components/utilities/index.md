# Utilities

Helper components for field manipulation and optimization.

## Overview

The `Utilities` module provides:
- **Pad/crop operators** for memory-efficient propagation
- **Tilt anchors** for off-axis beam tracking
- **Basis projection** for reduced-parameter optimization

## Examples

### Pad and Crop

```@example utilities
using FluxOptics

u = ScalarField(ones(ComplexF64, 128, 128), (2.0, 2.0), 1.064)

# Basic padding
u_pad = pad(u.electric, (256, 256))
size(u_pad)
```

```@example utilities
# Centered padding
offset = ((256-128)÷2, (256-128)÷2)
u_centered = pad(u.electric, (256, 256); offset=offset)

# Cropping back
u_crop = crop(u_centered, (128, 128); offset=offset)
size(u_crop)
```

### PadCropOperator

```@example utilities
# Efficient pad/crop in systems
u = ScalarField(ones(ComplexF64, 128, 128), (2.0, 2.0), 1.064)
u_tmp = zeros(ComplexF64, 256, 256)

pad_op = PadCropOperator(u, u_tmp; store_ref=true)
crop_op = adjoint(pad_op)

# Create source
source = ScalarSource(u)

# Avoid aliasing during propagation
prop = ASProp(set_field_data(u, u_tmp), 1000.0)
system = source |> pad_op |> prop |> crop_op

# Execute system
result = system()
size(result.out)
```

### Basis Projection

```@example utilities
using FluxOptics

u = ScalarField(ones(ComplexF64, 128, 128), (2.0, 2.0), 1.064)
source = ScalarSource(u)

# Polynomial basis
n_basis = 10
basis = make_spatial_basis((x, y, n) -> (x^2 + y^2)^n, 
                          (128, 128), (2.0, 2.0), 0:n_basis-1)

# Wrap trainable phase
phase = Phase(u, (x, y) -> 0.0; trainable=true)
wrapper = BasisProjectionWrapper(phase, basis, zeros(n_basis))

propagator = ASProp(u, 500.0)

# Optimize n_basis coefficients instead of 128×128 pixels
system = source |> wrapper |> propagator
result = system()

# Show we're optimizing only n_basis parameters
params = trainable(wrapper)
length(params.proj_coeffs)
```

## Key Types

- [`PadCropOperator`](@ref): Reversible pad/crop component
- [`TiltAnchor`](@ref): Tilt reference anchor
- [`BasisProjectionWrapper`](@ref): Basis coefficient optimization

## Key Functions

- [`pad`](@ref), [`crop`](@ref): Array padding and cropping
- [`make_spatial_basis`](@ref): Spatial basis generation
- [`make_fourier_basis`](@ref): Fourier basis generation

## See Also

- [Core](../core/index.md) for component interface
- [System](../system/index.md) for component composition

## Index

```@index
Modules = [FluxOptics.OpticalComponents]
Pages = ["utilities.md"]
Order = [:type, :function]
```
