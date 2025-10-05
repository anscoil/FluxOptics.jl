# Fields

The `ScalarField` type and field operations.

## Overview

The `Fields` module provides:
- **ScalarField type**: Central data structure for optical fields
- **Multi-wavelength support**: Handle multiple wavelengths simultaneously
- **Tilt tracking**: Manage off-axis propagation
- **Field operations**: Power, intensity, phase, normalization

## Quick Example

```julia
using FluxOptics

# Create a field
data = ones(ComplexF64, 128, 128, 3)  # 3 modes
u = ScalarField(data, (2.0, 2.0), [0.8, 1.064, 1.55])

# Field operations
P = power(u)              # Power per mode
I = intensity(u)          # Total intensity
normalize_power!(u)       # Normalize to unit power
φ = phase(u)              # Phase distribution

# Access and modify
u[:, :, 1] .= 0.0        # Clear first mode
u_copy = copy(u)         # Independent copy
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
