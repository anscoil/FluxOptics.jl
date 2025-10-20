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

### BasisProjectionWrapper

**Use cases:**
- Inverse problems with ill-posed pixel-wise optimization
- Enforcing smoothness constraints

**Basis selection:**
- Polynomials: smooth, global features
- Zernike: optical aberrations
- Fourier: periodic structures
- Custom: problem-specific functions

**Performance:**
- Optimization on N_basis coefficients vs N_pixels
- Typical: N_basis ≪ N_pixels (10-100 vs 10⁴-10⁶)

## See Also

- [Fields](../../fields/index.md) for ScalarField operations
- [Modulators](../modulators/index.md) for components to wrap
- [System](../system/index.md) for composition
