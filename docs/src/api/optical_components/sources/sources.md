# Sources API

```@meta
CurrentModule = FluxOptics.OpticalComponents
```

## Types

```@docs
ScalarSource
```

## Functions

```@docs
get_source(::ScalarSource)
```

## Usage Examples

### Creating Sources

```julia
using FluxOptics

# Define spatial grid and mode
xv, yv = spatial_vectors(64, 64, 2.0, 2.0)
gaussian = Gaussian(10.0)  # 10 μm waist

# Create field template
u0 = ScalarField(gaussian(xv, yv), (2.0, 2.0), 1.064)

# Static source
source_static = ScalarSource(u0)

# Trainable source for optimization
source_train = ScalarSource(u0; trainable=true, buffered=true)
```

### Multi-Mode Sources

```julia
# Create multi-mode field
modes = hermite_gaussian_groups(15.0, 3)  # HG modes up to order 2
mode_data = zeros(ComplexF64, 64, 64, length(modes))
for (i, mode) in enumerate(modes)
    mode_data[:, :, i] .= mode(xv, yv)
end

u_multi = ScalarField(mode_data, (2.0, 2.0), 1.064)
source_multi = ScalarSource(u_multi; trainable=true)
```

### In Optical Systems

```julia
# Build system with source
source = ScalarSource(u0; trainable=true)
phase = Phase(u0, (x, y) -> 0.0; trainable=true)
propagator = ASProp(u0, 1000.0)

system = source |> phase |> propagator

# Forward propagation
result = system()

# Extract optimized source after training
optimized_beam = get_source(source)
```

### Source Optimization

```julia
using Zygote

# Target field
target_mode = LaguerreGaussian(15.0, 0, 1)
target = ScalarField(target_mode(xv, yv), (2.0, 2.0), 1.064)

# Build system
source = ScalarSource(u0; trainable=true, buffered=true)
propagator = ASProp(u0, 500.0)
system = source |> propagator

# Define loss
metric = PowerCoupling(target)
loss() = 1.0 - metric(system())[]

# Optimize
params = trainable(system)
for iter in 1:100
    l, grads = withgradient(loss, params)
    # ... update params with optimizer ...
end

# Retrieve optimized beam
optimized = get_source(source)
```

## Technical Notes

### Multi-Wavelength Sources
Source fields support multiple wavelengths:
```julia
u0 = ScalarField(data, (2.0, 2.0), [0.8, 1.064, 1.55])
source = ScalarSource(u0; trainable=true)
# Optimizes all wavelength channels simultaneously
```

### Data Access
- `source.u0`: Access internal field (read/modify)
- `get_source(source)`: Safe copy extraction
- `get_data(source)`: Direct access to electric field array

## Design Considerations

### When to Use Static Sources
- Fixed beam profiles
- Reference fields
- Post-optimization analysis
- When gradient tracking is unnecessary

### When to Use Trainable Sources
- Inverse characterization of beam profiles
- Joint optimization with other components
- Adaptive optics applications

### Source Composition
Sources can represent:
- Single-mode beams (Gaussian, HG, LG)
- Multi-mode superpositions
- Arbitrary field distributions
- Experimentally measured fields

## Common Patterns

### Initializing from Modes
```julia
# Single mode
gaussian = Gaussian(20.0)
u0 = ScalarField(gaussian(xv, yv), (2.0, 2.0), 1.064)
source = ScalarSource(u0)

# Mode superposition
modes = [Gaussian(20.0), HermiteGaussian(20.0, 1, 0)]
coeffs = [1.0 + 0.0im, 0.5 + 0.3im]
field_data = sum(c * m(xv, yv) for (c, m) in zip(coeffs, modes))
u0 = ScalarField(field_data, (2.0, 2.0), 1.064)
source = ScalarSource(u0; trainable=true)
```

### Reusing Source Templates
```julia
# Create template
u_template = ScalarField((128, 128), (2.0, 2.0), 1.064)

# Multiple sources from template
source1 = ScalarSource(u_template; trainable=true)
source2 = ScalarSource(u_template; trainable=true)
# Independent optimization of each source
```

### Extracting Results
```julia
# After optimization
optimized_field = get_source(source)
power_per_mode = power(optimized_field)
phase_profile = phase(optimized_field)

# Save for later use
source_static = ScalarSource(optimized_field)
```

## See Also

- [`ScalarField`](@ref) for field representation
- [`propagate`](@ref) for field generation
- [Modes](../../modes/index.md) for beam profile generation
- [System](../system/index.md) for optical system composition
