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

## Technical Notes

### Angular Spectrum Method

Kernel in frequency domain:
- On-axis: `H = exp(i 2π z √(1/λ² - fx² - fy²))`
- Tilted: Modified with tilt offsets

Handles both paraxial and non-paraxial regimes.

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

- [`propagate`](@ref) for field propagation
- [Bulk Propagators](../bulk/index.md) for inhomogeneous media
- [Core](../core/index.md) for component interface
