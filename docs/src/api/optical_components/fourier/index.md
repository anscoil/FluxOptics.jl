# Fourier

Fourier-domain operations and filtering.

## Overview

The `Fourier` module provides:
- **Fourier domain wrappers** for spatial components
- **Frequency-domain phase masks** for filtering
- **Frequency-domain amplitude masks** for spectral shaping
- **FFT/IFFT operators** for domain transformations

## Quick Example

```julia
using FluxOptics

u = ScalarField(ones(ComplexF64, 256, 256), (1.0, 1.0), 1.064)

# Low-pass filter in Fourier domain
f_cutoff = 0.2  # 1/μm
lowpass = FourierMask(u, (fx, fy) -> sqrt(fx^2 + fy^2) < f_cutoff ? 1.0 : 0.0)

# Frequency-domain phase (e.g., propagation kernel)
fourier_phase = FourierPhase(u, (fx, fy) -> π * (fx^2 + fy^2))

# Apply spatial component in Fourier domain
phase_spatial = Phase(u, (x, y) -> 0.01*x^2)
phase_freq = FourierWrapper(u, phase_spatial)

# Use in system
system = ScalarSource(u) |> lowpass |> fourier_phase
```

## Key Types

- [`FourierOperator`](@ref): FFT/IFFT transformation
- [`FourierWrapper`](@ref): Apply component in Fourier domain
- [`FourierPhase`](@ref): Phase mask in Fourier space
- [`FourierMask`](@ref): Amplitude mask in Fourier space

## See Also

- [Modulators](../modulators/index.md) for spatial-domain modulation
- [Core](../core/index.md) for component interface

## Index

```@index
Modules = [FluxOptics.OpticalComponents]
Pages = ["fourier.md"]
Order = [:type, :function]
```
