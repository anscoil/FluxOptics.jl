# Fields

The `ScalarField` type and field operations.

## Overview

The `Fields` module provides:
- **ScalarField type**: Central data structure for optical fields
- **Multi-wavelength support**: Handle multiple wavelengths simultaneously
- **Tilt tracking**: Manage off-axis propagation
- **Field operations**: Power, intensity, phase, normalization

## Examples

### Creating Fields

```@example fields
using FluxOptics

# Create field from mode
gaussian = Gaussian(20.0)
xv, yv = spatial_vectors(128, 128, 2.0, 2.0)
field_data = gaussian(xv, yv)
u = ScalarField(field_data, (2.0, 2.0), 1.064)

size(u)  # Grid dimensions
```

### Field Operations

```@example fields
# Power calculation and normalization
P_initial = power(u)[]  # Extract scalar from 0-dimensional array

normalize_power!(u, 2.5)  # Normalize to 2.5 W
P_normalized = power(u)[]

(P_initial, P_normalized)  # Before: ~1.0, After: 2.5
```

```@example fields
# Intensity and phase distributions
I = intensity(u)  # 2D array: total intensity
φ = phase(u)      # Same size as u: phase at each point

(size(I), size(φ))
```

### Field Manipulation

```@example fields
# Copy and fill operations
u_copy = copy(u)          # Independent copy
u_tmp = similar(u)        # Allocate uninitialized

fill!(u_tmp, 1.0 + 0.0im) # Fill with constant value

# Update field data
new_data = 0.5 .* u.electric
u_new = set_field_data(u, new_data)

power(u_new)[]  # Power scaled by 0.5² = 0.25
```

### Multi-Wavelength Fields

```@example fields
# Multiple wavelengths simultaneously
λs = [0.8, 1.064, 1.55]
data_multi = zeros(ComplexF64, 128, 128, 3)
for (i, λ) in enumerate(λs)
    g = Gaussian(20.0)
    data_multi[:, :, i] .= g(xv, yv)
end

u_multi = ScalarField(data_multi, (2.0, 2.0), λs)

# Power per wavelength
power(u_multi)  # Returns 1×1×3 array
```

### Field Comparison

```@example fields
# Compare with another mode
hg10 = HermiteGaussian(20.0, 1, 0)
u2 = ScalarField(hg10(xv, yv), (2.0, 2.0), 1.064)
normalize_power!(u2)

# Coupling efficiency (mode overlap)
η = coupling_efficiency(u, u2)

η[]  # Overlap between Gaussian and HG10
```

### Field with Tilts

```@example fields
# Off-axis propagation
u_tilted = ScalarField(field_data, (2.0, 2.0), 1.064; tilts=(0.01, 0.005))

is_on_axis(u_tilted)  # false
```

## Key Types

- [`ScalarField`](@ref): Optical field with grid information

## Key Functions

**Field Construction and Data**
- [`ScalarField`](@ref): Create optical field
- [`set_field_data`](@ref): Update field data
- [`similar`](@ref similar(::ScalarField)): Allocate similar field
- [`copy`](@ref copy(::ScalarField)): Copy field
- [`fill!`](@ref fill!(::ScalarField, ::Any)): Fill field with values
- [`copyto!`](@ref copyto!(::ScalarField, ::ScalarField)): Copy between fields
- [`collect`](@ref collect(::ScalarField)): Convert to CPU array

**Field Properties**
- [`size`](@ref size(::ScalarField)): Get field dimensions
- [`ndims`](@ref ndims(::ScalarField)): Number of dimensions
- [`eltype`](@ref eltype(::ScalarField)): Element type

**Tilts Management**
- [`set_field_tilts`](@ref): Update field tilts
- [`offset_tilts!`](@ref): Add tilt offset with phase correction
- [`is_on_axis`](@ref): Check if field is on-axis

**Power and Intensity**
- [`power`](@ref): Calculate field power
- [`normalize_power!`](@ref): Normalize to target power
- [`intensity`](@ref): Calculate total intensity
- [`phase`](@ref): Extract phase distribution

**Field Comparison**
- [`coupling_efficiency`](@ref): Coupling efficiency between fields
- [`dot`](@ref dot(::ScalarField, ::ScalarField)): Field overlap integral

**Vectorization**
- [`vec`](@ref vec(::ScalarField)): Vectorize into independent ScalarFields

**Broadcasting and Indexing**
- [`broadcasted`](@ref Base.Broadcast.broadcasted(::Function, ::ScalarField)): Element-wise operations
- [`getindex`](@ref getindex(::ScalarField, ::Any)): Access field elements
- [`conj`](@ref conj(::ScalarField)): Complex conjugate

## See Also

- [Modes](../modes/index.md) for generating optical modes
- [GridUtils](../gridutils/index.md) for coordinate systems

## Index

```@index
Modules = [FluxOptics.Fields]
Order = [:type, :function]
```
