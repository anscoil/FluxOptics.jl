# Utilities

Helper components for field manipulation and optimization.

## Overview

The `Utilities` module provides:
- **Pad/crop operators** for memory-efficient propagation
- **Tilt anchors** for off-axis beam tracking
- **Basis projection** for reduced-parameter optimization

## Quick Example

```julia
using FluxOptics

u = ScalarField(ones(ComplexF64, 128, 128), (2.0, 2.0), 1.064)

# Pad for propagation, then crop back
u_tmp = zeros(ComplexF64, 256, 256)
pad_op = PadCropOperator(u, u_tmp; store_ref=true)
crop_op = adjoint(pad_op)

system = source |> pad_op |> propagator |> crop_op

# Tilt anchor for off-axis systems
anchor = TiltAnchor(u)
system = source |> anchor |> components...

# Basis projection for regularization
basis = make_spatial_basis((x, y, n) -> x^n * y^n, (128, 128), (2.0, 2.0), 0:5)
phase = Phase(u, (x, y) -> 0.0; trainable=true)
wrapper = BasisProjectionWrapper(phase, basis, zeros(6))
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
