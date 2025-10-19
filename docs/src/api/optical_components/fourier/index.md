# Fourier

Fourier-domain operations and filtering.

## Overview

The `Fourier` module provides tools for applying operations in the frequency domain, including spatial filtering, frequency-domain masks, and wrapping spatial components to operate on Fourier transforms.

## Quick Example

### Frequency-Domain Filtering

```@example fourier
using FluxOptics

xv, yv = spatial_vectors(128, 128, 1.0, 1.0)
u = ScalarField(Gaussian(20.0)(xv, yv), (1.0, 1.0), 1.064)

# Sharp low-pass filter
f_cutoff = 0.02  # 1/μm
lowpass = FourierMask(u, (fx, fy) -> sqrt(fx^2 + fy^2) < f_cutoff ? 1.0 : 0.0)

# Apply filter
u_filtered = propagate(u, lowpass, Forward)

# Check power conservation (should be < 1 due to filtering)
power(u_filtered)[] / power(u)[]
```

```@example fourier
# Gaussian filter (soft cutoff)
sigma_f = 0.15  # 1/μm
gaussian_filter = FourierMask(u, (fx, fy) -> exp(-(fx^2 + fy^2)/(2*sigma_f^2)))

u_smooth = propagate(u, gaussian_filter, Forward)

power(u_smooth)[] / power(u)[]
```

### Frequency-Domain Phase

```@example fourier
# Parabolic phase in frequency (quadratic chirp in spatial domain)
fourier_phase = FourierPhase(u, (fx, fy) -> π * 0.01 * (fx^2 + fy^2))

u_chirped = propagate(u, fourier_phase, Forward)

# Phase modulation doesn't affect power
power(u_chirped)[]
```

### Wrapping Spatial Components

```@example fourier
# Apply spatial phase mask in Fourier domain
phase_spatial = Phase(u, (x, y) -> 0.01 * x^2; trainable=true)
phase_in_fourier = FourierWrapper(u, phase_spatial)

# Equivalent to: FFT → phase_spatial → IFFT
istrainable(phase_in_fourier)
```

### Using in Systems

```@example fourier
# Combine spatial and frequency filtering
source = ScalarSource(u)
lowpass = FourierMask(u, (fx, fy) -> sqrt(fx^2 + fy^2) < 0.02 ? 1.0 : 0.0)
phase = Phase(u, (x, y) -> 0.01 * (x^2 + y^2))
prop = ASProp(u, 1000.0)

system = source |> lowpass |> phase |> prop

result = system()
power(result.out)[]
```

## Key Types

- [`FourierOperator`](@ref): FFT/IFFT transformation (low-level)
- [`FourierWrapper`](@ref): Apply component in Fourier domain (FFT → component → IFFT)
- [`FourierPhase`](@ref): Phase mask in Fourier space (convenience)
- [`FourierMask`](@ref): Amplitude/complex mask in Fourier space (convenience)

## Coordinate Convention

Functions receive **frequency arguments** `(fx, fy)` in units of **1/length**:
- Frequency grid: `fx = fftfreq(nx, 1/dx)`
- Zero frequency at center after `fftshift`

Example: For `dx = 1.0 μm` and `nx = 128`, `fx` ranges from -0.5 to +0.5-1/128 μm⁻¹.

## Component Relationships

### FourierPhase and FourierMask
Convenience constructors that internally use `FourierWrapper`:
```julia
# These are equivalent:
FourierPhase(u, (fx, fy) -> φ(fx, fy))
FourierWrapper(u, Phase(u, ...))  # Phase evaluated in frequency

FourierMask(u, (fx, fy) -> m(fx, fy))
FourierWrapper(u, Mask(u, ...))   # Mask evaluated in frequency
```

### FourierWrapper
General wrapper that applies: `u → IFFT[component(FFT[u])]`

Useful for applying spatial-domain components to frequency content.

## See Also

- [Modulators](../modulators/index.md) - Spatial-domain phase and amplitude modulation
- [Core](../core/index.md) - Trainability and component interface
- [Free-Space Propagators](../freespace/index.md) - All propagation methods use `FourierWrapper`

## Index

```@index
Modules = [FluxOptics.OpticalComponents]
Pages = ["fourier.md"]
Order = [:type, :function]
```
