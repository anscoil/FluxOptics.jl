# System API

```@meta
CurrentModule = FluxOptics.OpticalComponents
```

## Types

```@docs
OpticalSystem
AbstractSequence
OpticalSequence
FieldProbe
```

## Functions

```@docs
get_source(::OpticalSystem)
get_components
get_sequence
```

## Technical Notes

### Component Merging Rules

When `merge_components=true`, compatible adjacent components are merged:

**Phase + Phase:**
```julia
phase1 = Phase(u, (x, y) -> x^2)
phase2 = Phase(u, (x, y) -> y^2)
# Merged: φ_total = φ₁ + φ₂
```

**Mask × Mask:**
```julia
mask1 = Mask(u, (x, y) -> 0.8)
mask2 = Mask(u, (x, y) -> 0.9)
# Merged: m_total = m₁ × m₂
```

**Mask × Phase → Mask:**
```julia
mask = Mask(u, (x, y) -> 0.8)
phase = Phase(u, (x, y) -> π/4)
# Merged: m_total = m × exp(iφ)
```

**FourierOperator + FourierOperator:**
```julia
fft = FourierOperator(u, true)
ifft = FourierOperator(u, false)
# Merged: cancels to identity (removed)
```

**Requirements for merging:**
- Components must be `Static` (non-trainable)
- Arrays must have compatible sizes
- Components must be adjacent in sequence

### Pipe Operator Behavior

The pipe operator `|>` creates `OpticalSystem` instances:

```julia
# Source + Component → OpticalSystem
system = source |> phase

# Component + Component → OpticalSystem (no source)
sequence_system = phase |> lens

# System + Component → OpticalSystem
system = system |> propagator

# System + NamedTuple → OpticalSystem (update options)
system = system |> (; merge_components=true)
```

### Execution and Output

System execution returns a NamedTuple:
```julia
result = system()
# result.out: final field
# result.probes: IdDict{FieldProbe, ScalarField}
```

For systems without probes:
```julia
result = system()
output = result.out  # Access final field
```

### In-Place Mode

When `inplace=true`:
- Fields are modified during propagation
- Reduces memory allocations
- Compatible with probes (probes always copy)

```julia
system_inplace = source |> phase |> lens |> (; inplace=true)
result = system_inplace()
```

## See Also

- [Core](../core/index.md) for component interface
- [Sources](../sources/index.md) for source components
- [Modulators](../modulators/index.md) for modulation components
