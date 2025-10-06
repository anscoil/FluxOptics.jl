# Modulators API

```@meta
CurrentModule = FluxOptics.OpticalComponents
```

## Types

```@docs
Phase
Mask
TeaDOE
TeaReflector
```

## Usage Examples

### Phase Masks

```julia
using FluxOptics

u = ScalarField(ones(ComplexF64, 256, 256), (2.0, 2.0), 1.064)
xv, yv = spatial_vectors(256, 256, 2.0, 2.0)

# Parabolic lens
phase_lens = Phase(u, (x, y) -> π/(1000^2) * (x^2 + y^2))

# Trainable phase mask
phase_opt = Phase(u, (x, y) -> 0.0; trainable=true, buffered=true)

# From array
phase_data = load_phase_profile(...)
phase_measured = Phase(u, phase_data)
```

### Amplitude Masks

```julia
# Circular aperture
radius = 50.0
aperture = Mask(u, (x, y) -> sqrt(x^2 + y^2) < radius ? 1.0 : 0.0)

# Gaussian apodization
w0 = 40.0
apodization = Mask(u, (x, y) -> exp(-(x^2 + y^2)/(2*w0^2)))

# Trainable complex mask
mask_opt = Mask(u, (x, y) -> 1.0 + 0.0im; trainable=true)
```

### Diffractive Elements

```julia
# Sinusoidal grating
Δn = 0.5
period = 50.0
grating = TeaDOE(u, Δn, (x, y) -> 0.5 * sin(2π * x / period))

# Binary phase grating
grating_binary = TeaDOE(u, 0.5, (x, y) -> mod(x, 20) < 10 ? 1.0 : 0.0)

# Trainable DOE
doe_opt = TeaDOE(u, 0.5, (x, y) -> 0.0; trainable=true, buffered=true)
```

### Reflective Elements

```julia
# Deformable mirror
dm = TeaReflector(u, (x, y) -> 0.0; trainable=true)

# Parabolic mirror
mirror = TeaReflector(u, (x, y) -> 0.001 * (x^2 + y^2))

# Mirror with coating reflectivity
r_coating = λ -> 0.95  # 95% reflective
mirror_coated = TeaReflector(u, (x, y) -> 0.0; r=r_coating)
```

## Technical Notes

### Phase Mask Behavior
- Applies `u → u × exp(iφ(x,y))`
- Direction-dependent: Forward uses `+φ`, Backward uses `-φ`
- Trainable parameter: phase array `ϕ`

### Mask Transmission
- Applies `u → u × m(x,y)` where `m` is complex
- Real values: amplitude modulation
- Complex values: joint amplitude and phase
- Direction-dependent: Forward uses `m`, Backward uses `conj(m)`

### TeaDOE Physics
- Phase shift: `φ = 2π Δn h(x,y) / λ`
- Wavelength-dependent when using multiple wavelengths
- Optional reflectivity function `r(λ)`
- Trainable parameter: height profile `h`

### TeaReflector
- Special case: `Δn = 2` (reflection doubles path)
- Phase shift: `φ = 4π h(x,y) / λ`
- Models deformable mirrors, curved mirrors

## See Also

- [FourierPhase](@ref) for frequency-domain phase modulation
- [FourierMask](@ref) for frequency-domain amplitude modulation
- [Core](../core/index.md) for trainability system
