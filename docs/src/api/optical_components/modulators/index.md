# Modulators

Phase and amplitude modulation components.

## Overview

The `Modulators` module provides components that modify the amplitude and phase of optical fields through masks, diffractive elements, and reflective surfaces.

## Quick Example

### Phase Modulation

```@example modulators
using FluxOptics

xv, yv = spatial_vectors(128, 128, 2.0, 2.0)
u = ScalarField(Gaussian(50.0)(xv, yv), (2.0, 2.0), 1.064)

# Parabolic phase (lens-like)
phase_lens = Phase(u, (x, y) -> π/(1000^2) * (x^2 + y^2))

istrainable(phase_lens)  # false (static)
```

```@example modulators
# Trainable phase mask for optimization
phase_opt = Phase(u, (x, y) -> 0.0; trainable=true, buffered=true)

(istrainable(phase_opt), isbuffered(phase_opt))
```

### Amplitude Modulation

```@example modulators
# Circular aperture
radius = 80.0
aperture = Mask(u, (x, y) -> sqrt(x^2 + y^2) < radius ? 1.0 : 0.0)

# Apply to field
u_masked = propagate(u, aperture, Forward)

# Check power reduction
(power(u)[], power(u_masked)[])
```

```@example modulators
# Gaussian apodization (smooth amplitude taper)
w0 = 100.0
apodization = Mask(u, (x, y) -> exp(-(x^2 + y^2)/(2*w0^2)))

u_apodized = propagate(u, apodization, Forward)

power(u_apodized)[]
```

### Diffractive Elements

```@example modulators
# Sinusoidal grating
Δn = 0.5  # Refractive index difference
period = 50.0  # Grating period
grating = TeaDOE(u, Δn, (x, y) -> 0.5 * sin(2π * x / period))

# Trainable DOE for beam shaping
doe_opt = TeaDOE(u, 0.5, (x, y) -> 0.0; trainable=true, buffered=true)

istrainable(doe_opt)
```

### Using in Systems

```@example modulators
# Combine multiple modulators
source = ScalarSource(u)
phase = Phase(u, (x, y) -> 0.01 * (x^2 + y^2))
aperture = Mask(u, (x, y) -> sqrt(x^2 + y^2) < 60.0 ? 1.0 : 0.0)
prop = ASProp(u, 1000.0)

system = source |> phase |> aperture |> prop

result = system()
power(result.out)[]
```

## Key Types

- [`Phase`](@ref): Pure phase modulation exp(iφ)
- [`Mask`](@ref): Amplitude or complex transmission
- [`TeaDOE`](@ref): Diffractive element (thin element approximation)
- [`TeaReflector`](@ref): Reflective element (mirrors, deformable mirrors)

## Physical Behavior

### Phase Masks
- Apply transmission: `t(x,y) = exp(iφ(x,y))`
- Forward: `+φ`, Backward: `-φ` (adjoint)
- Conserve power (unitary)

### Amplitude Masks
- Apply transmission: `t(x,y) = m(x,y)` (complex-valued)
- Forward: `m`, Backward: `conj(m)` (adjoint)
- Real `m`: pure amplitude, Complex `m`: amplitude + phase

### Diffractive Elements (TeaDOE)
- Phase from surface height: `φ = 2π Δn h(x,y) / λ`
- Wavelength-dependent (dispersion)
- Optional wavelength-dependent reflectivity

### Reflectors (TeaReflector)
- Special case: `Δn = 2` (reflection doubles optical path)
- Phase: `φ = 4π h(x,y) / λ`
- Models deformable mirrors, curved mirrors

## See Also

- [Typical Workflow](../../index.md#typical-workflow-beam-splitter) - Phase mask optimization example
- [Core](../core/index.md) - Trainability system
- [Fourier](../fourier/index.md) - Frequency-domain modulation

## Index

```@index
Modules = [FluxOptics.OpticalComponents]
Pages = ["modulators.md"]
Order = [:type, :function]
```
