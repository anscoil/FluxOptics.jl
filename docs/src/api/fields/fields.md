# Fields API

```@meta
CurrentModule = FluxOptics.Fields
```

## ScalarField Type

```@docs
ScalarField
```

### Examples

```julia
using FluxOptics

# From existing data
data = rand(ComplexF64, 64, 64)
u = ScalarField(data, (2.0, 2.0), 1.064)

# Zero-initialized
u = ScalarField((128, 128), (1.5, 1.5), 1.064)

# Multi-mode field
data = zeros(ComplexF64, 128, 128, 5)
u = ScalarField(data, (2.0, 2.0), 1.064)

# Multi-wavelength field
λs = [0.8, 1.064, 1.55]
data = zeros(ComplexF64, 64, 64, 3)
u = ScalarField(data, (1.0, 1.0), λs)

# Field with tilts
u = ScalarField(data, (2.0, 2.0), 1.064; tilts=(0.01, 0.005))
```

## Field Data Management

```@docs
set_field_data
similar(::ScalarField)
copy(::ScalarField)
fill!(::ScalarField, ::Number)
copyto!(::ScalarField, ::ScalarField)
collect(::ScalarField)
```

### Examples

```julia
# Update field data
new_data = rand(ComplexF64, size(u)...)
u_new = set_field_data(u, new_data)

# Copy and allocate
u_copy = copy(u)
u_tmp = similar(u)

# Fill operations
fill!(u, 1.0 + 0.0im)
fill!(u, rand(ComplexF64, size(u)...))

# Copy between fields
copyto!(u_tmp, u)

# Convert GPU to CPU array
data = collect(u)  # Returns Array
```

## Field Properties

```@docs
size(::ScalarField)
ndims(::ScalarField)
eltype(::ScalarField)
```

### Examples

```julia
# Get dimensions
nx, ny = size(u)[1:2]
n_modes = size(u, 3)

# Number of dimensions
ndims(u)           # Total dimensions
ndims(u, true)     # Spatial dimensions only

# Element type
eltype(u)  # ComplexF64 or ComplexF32
```

## Tilt Management

```@docs
set_field_tilts
offset_tilts!
is_on_axis
```

### Examples

```julia
# Set new tilts
u_tilted = set_field_tilts(u, (0.02, 0.015))

# Add tilt offset (modifies tilts and applies phase)
offset_tilts!(u, (0.01, 0.005))

# Check if on-axis
is_on_axis(u)  # true/false
```

## Power and Intensity

```@docs
power
normalize_power!
intensity
phase
```

### Examples

```julia
# Single mode field
u = ScalarField(ones(ComplexF64, 128, 128), (2.0, 2.0), 1.064)
P = power(u)  # Returns 1×1 array

# Normalize to 1 W
normalize_power!(u)
@assert isapprox(power(u)[], 1.0)

# Normalize to specific power
normalize_power!(u, 1e-3)  # 1 mW

# Multi-mode field
data = rand(ComplexF64, 64, 64, 3)
u = ScalarField(data, (1.0, 1.0), 1.064)
P = power(u)  # Returns 1×1×3 array

# Different power per mode
target_powers = reshape([1.0, 1.5, 2.0], 1, 1, 3)
normalize_power!(u, target_powers)

# Total intensity (sum over all modes)
I_total = intensity(u)  # Returns 2D array

# Phase extraction
φ = phase(u)  # Returns array same size as u
```

## Field Comparison

```@docs
coupling_efficiency
dot(::ScalarField)
```

### Examples

```julia
# Modal overlap integral
u1 = ScalarField(gaussian1(xv, yv), (2.0, 2.0), 1.064)
u2 = ScalarField(gaussian2(xv, yv), (2.0, 2.0), 1.064)

overlap = dot(u1, u2)  # Complex overlap
η = coupling_efficiency(u1, u2)  # Real, between 0 and 1

# Multi-mode fields
data1 = rand(ComplexF64, 64, 64, 3)
data2 = rand(ComplexF64, 64, 64, 3)
u1 = ScalarField(data1, (2.0, 2.0), 1.064)
u2 = ScalarField(data2, (2.0, 2.0), 1.064)

overlaps = dot(u1, u2)  # 3-element array
efficiencies = coupling_efficiency(u1, u2)  # 3-element array
```

## Vectorization

```@docs
vec(::ScalarField)
```

### Examples

```julia
# Vectorize into individual ScalarFields
u_vec = vec(u)  # Vector of ScalarField objects

# Iterate over modes
for (i, mode_field) in enumerate(u_vec)
    P_mode = power(mode_field)
    I_mode = intensity(mode_field)
    println("Mode $i: Power = $(P_mode[])")
end

# Collect results
powers = [power(mode)[] for mode in u_vec]
peak_intensities = [maximum(intensity(mode)) for mode in u_vec]
```

## Broadcasting and Indexing

```@docs
Base.Broadcast.broadcasted(::Function, ::ScalarField)
getindex(::ScalarField, ::Any)
conj(::ScalarField)
```

### Examples

```julia
# Broadcasting
u_scaled = 2.0 .* u
u_shifted = u .+ 1.0

# Indexing (returns view)
mode_1 = u[:, :, 1]
row = u[32, :, :]

# Complex conjugate
u_conj = conj(u)
```

## Technical Notes

### Data Layout
- First `Nd` dimensions: spatial (x, y for 2D)
- Remaining dimensions: modes, wavelengths, etc.
- Contiguous in memory for performance

### Wavelength Handling
- Single wavelength: stored as scalar, broadcast to field
- Multiple wavelengths: array broadcasts with extra (non-spatial) dimensions

### Tilt Handling
- Tilts are passed as `(θx, θy)` tuples (scalars or arrays) to the constructor
- Internally stored as tuple of arrays for in-place modification
- Each tilt component broadcasts with extra (non-spatial) dimensions
- Tilts represent Fourier offset: `fx₀ = sin(θx)/λ`

### Power Calculation
- Power = ∫∫ |u(x,y)|² dx dy
- Numerical integration: `P ≈ Σᵢⱼ |u[i,j]|² × dx × dy`
- Units depend on field amplitude units and spatial units

### Intensity vs Power
- **Intensity**: |u(x,y)|² (per unit area)
- **Power**: Spatial integral of intensity
- For multi-mode: intensity sums over modes, power is per-mode

### Coupling Efficiency
- η = |⟨u₁,u₂⟩|² / (‖u₁‖ ‖u₂‖)
- Always real, between 0 and 1
- Insensitive to global phase
- Equals 1 for identical fields

### Memory Considerations
- `copy()` creates independent copy of data
- `similar()` allocates uninitialized memory
- `set_field_data()` creates new field using provided data and same other parameters
- Use views for slicing without copying

## Performance Tips

1. **Pre-allocate**: Use `similar()` for temporary fields
2. **In-place operations**: Modify `.electric` directly when possible
3. **Avoid unnecessary copies**: Use `set_field_data()` or views
4. **Vectorization**: Use `vec()` only when needed for iteration
5. **GPU**: Field data can be moved to GPU with `CUDA.cu(u)`

## See Also

- [Modes](../modes/index.md) for generating field data
- [GridUtils](../gridutils/index.md) for coordinate systems
