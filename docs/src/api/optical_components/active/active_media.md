# Active Media API

```@meta
CurrentModule = FluxOptics.OpticalComponents
```

## Types

```@docs
GainSheet
```

## Usage Examples

### Spatially-Varying Gain

```julia
# Gaussian pump profile
w_pump = 100.0  # Pump beam waist
gain_gaussian = GainSheet(u, 0.1, 1e6, (x, y) -> 2.0 * exp(-(x^2 + y^2)/(2*w_pump^2)))

# Top-hat pumped region
r_pump = 50.0
gain_tophat = GainSheet(u, 0.1, 1e6, (x, y) -> sqrt(x^2 + y^2) < r_pump ? 2.0 : 0.0)

# Custom profile
gain_custom = GainSheet(u, 0.1, 1e6, (x, y) -> 2.0 * (1 + 0.5*cos(2π*x/100)))
```

## Technical Notes

### Gain Saturation

Gain coefficient: `g(I) = g₀ / (1 + I/Isat)`

- Low intensity (I ≪ Isat): `g ≈ g₀` (linear gain)
- High intensity (I ≫ Isat): `g ≈ g₀ × Isat/I` (saturated)

### Transmission

Field after gain: `u_out = u_in × exp(g(I) × dz)`

- Amplitude gain: `|u_out| = |u_in| × exp(g(I) × dz / 2)`
- Power gain: `P_out = P_in × exp(g(I) × dz)`

### Saturation Effects

- High-intensity regions experience less gain
- Leads to beam profile distortion
- Important for high-power laser systems
- Can be used for pulse shaping

### Trainable Gain

When `trainable=true`, optimizes spatial gain profile `g₀(x,y)`

## See Also

- [Modulators](../modulators/index.md) for passive modulation
- [Core](../core/index.md) for trainability system
