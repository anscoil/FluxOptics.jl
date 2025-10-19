# Fields API

```@meta
CurrentModule = FluxOptics.Fields
```

## ScalarField Type

```@docs
ScalarField
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

## Field Properties

```@docs
size(::ScalarField)
ndims(::ScalarField)
eltype(::ScalarField)
```

## Tilt Management

```@docs
set_field_tilts
offset_tilts!
is_on_axis
```

## Power and Intensity

```@docs
power
normalize_power!
intensity
phase
```

## Field Comparison

```@docs
coupling_efficiency
dot(::ScalarField)
```

## Vectorization

```@docs
vec(::ScalarField)
```

## Broadcasting and Indexing

```@docs
Base.Broadcast.broadcasted(::Function, ::ScalarField)
getindex(::ScalarField, ::Any)
conj(::ScalarField)
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
