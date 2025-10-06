# Fourier API

```@meta
CurrentModule = FluxOptics.OpticalComponents
```

## Types

```@docs
FourierOperator
FourierWrapper
FourierPhase
FourierMask
```

## Usage Examples

### Fourier Domain Filtering

```julia
using FluxOptics

u = ScalarField(ones(ComplexF64, 256, 256), (1.0, 1.0), 1.064)

# Sharp low-pass filter
f_cutoff = 0.2  # 1/μm
lowpass = FourierMask(u, (fx, fy) -> sqrt(fx^2 + fy^2) < f_cutoff ? 1.0 : 0.0)

# Gaussian filter
sigma_f = 0.15
gaussian_filter = FourierMask(u, (fx, fy) -> exp(-(fx^2 + fy^2)/(2*sigma_f^2)))

# Band-pass filter
f_min, f_max = 0.1, 0.3
bandpass = FourierMask(u, (fx, fy) -> begin
    f = sqrt(fx^2 + fy^2)
    f_min < f < f_max ? 1.0 : 0.0
end)
```

### Fourier Domain Phase

```julia
# Parabolic phase in frequency
fourier_phase = FourierPhase(u, (fx, fy) -> 0.01*(fx^2 + fy^2))

# Soft aperture in frequency domain
sigma_f = 0.2
soft_aperture = FourierPhase(u, (fx, fy) -> -π*(fx^2 + fy^2)/(2*sigma_f^2))

# Trainable frequency filter
fourier_opt = FourierPhase(u, (fx, fy) -> 0.0; trainable=true, buffered=true)
```

### Wrapping Components

```julia
# Apply spatial phase mask in Fourier domain
phase_spatial = Phase(u, (x, y) -> 0.01*x^2; trainable=true)
phase_fourier = FourierWrapper(u, phase_spatial)

# Wrap multiple components
mask = Mask(u, (x, y) -> exp(-(x^2 + y^2)/100))
phase = Phase(u, (x, y) -> x*y*0.01)
wrapped = FourierWrapper(u, mask, phase)
```

### Manual FFT Operations

```julia
# Explicit Fourier transform
ft = FourierOperator(u, true)   # Forward FFT
ift = FourierOperator(u, false)  # Inverse FFT

# Apply operations in Fourier domain
u_freq = propagate(u, ft, Forward)
# ... modify u_freq ...
u_back = propagate(u_freq, ift, Forward)
```

## Technical Notes

### FourierWrapper Behavior
- Applies FFT → component → IFFT sequence
- Equivalent to: `u → IFFT[component(FFT[u])]`
- Useful for frequency-domain filtering with spatial components

### Coordinate Convention
- Functions receive frequency arguments: `(fx, fy)` in 1/length units
- Frequency grid: `fx = fftfreq(nx, 1/dx)`
- Zero frequency at center after `fftshift`

### FourierPhase vs FourierMask
- **FourierPhase**: Pure phase modulation in frequency (`exp(iφ(fx,fy))`)
- **FourierMask**: Complex transmission in frequency (`m(fx,fy)`)
- Both are internally `FourierWrapper(u, Phase/Mask(...))`

### Trainability
All Fourier components support trainability:
```julia
fourier_opt = FourierPhase(u, (fx, fy) -> 0.0; trainable=true, buffered=true)
# Optimizes phase in frequency domain
```

## See Also

- [Phase](@ref) for spatial-domain phase masks
- [Mask](@ref) for spatial-domain amplitude masks
- [Core](../core/index.md) for component interface
