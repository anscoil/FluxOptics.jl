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

## Usage Examples

### Building Systems

```julia
using FluxOptics

u = ScalarField(ones(ComplexF64, 256, 256), (2.0, 2.0), 1.064)

# Create components
source = ScalarSource(u; trainable=true)
phase = Phase(u, (x, y) -> 0.01*(x^2 + y^2); trainable=true)
lens = FourierLens(u, (2.0, 2.0), 1000.0)
prop = ASProp(u, 500.0)

# Build system with pipe operator
system = source |> phase |> lens |> prop

# Execute
result = system()
output = result.out
```

### Component Merging

```julia
# Without merging: 4 operations
phase1 = Phase(u, (x, y) -> x^2)
phase2 = Phase(u, (x, y) -> y^2)
phase3 = Phase(u, (x, y) -> x*y)
system = source |> phase1 |> phase2 |> phase3

# With merging: 2 operations (phases merged into one)
system_merged = source |> phase1 |> phase2 |> phase3 |> (; merge_components=true)

# Check merged components
components = get_components(system_merged)
length(components)  # Fewer than original
```

### Field Probes

```julia
# Insert probes at strategic locations
probe_after_phase = FieldProbe()
probe_after_lens = FieldProbe()

system = source |> 
         phase |> probe_after_phase |>
         lens |> probe_after_lens |>
         prop

# Execute and access intermediate fields
result = system()
field_after_phase = result.probes[probe_after_phase]
field_after_lens = result.probes[probe_after_lens]
final_field = result.out
```

### Sequences

```julia
# Create reusable sequence
mask = Mask(u, (x, y) -> exp(-(x^2 + y^2)/100))
phase = Phase(u, (x, y) -> 0.01*x*y)
sequence = OpticalSequence(mask, phase)

# Apply to field
u_out = propagate(u, sequence, Forward)

# Use in different systems
system1 = source1 |> sequence |> prop1
system2 = source2 |> sequence |> prop2
```

### System Configuration

```julia
# Forward propagation (default)
system_fwd = source |> phase |> prop

# Backward propagation
system_bwd = source |> phase |> prop |> (; direction=Backward)

# In-place mode (memory efficient)
system_inplace = source |> phase |> prop |> (; inplace=true)

# Combined options
system = source |> phase |> prop |> (; 
    direction=Forward, 
    inplace=false, 
    merge_components=true
)
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
- Use with caution: intermediate fields are lost
- Not compatible with probes (probes always copy)

```julia
system_inplace = source |> phase |> lens |> (; inplace=true)
result = system_inplace()  # Efficient but no intermediate fields
```

### Trainability Propagation

System trainability determined by components:
- Any trainable component → trainable system
- All static → static system
- Check with `istrainable(system)`

```julia
source = ScalarSource(u; trainable=true)
phase = Phase(u, (x, y) -> x^2)  # Static
system = source |> phase
istrainable(system)  # true (source is trainable)
```

## See Also

- [Core](../core/index.md) for component interface
- [Sources](../sources/index.md) for source components
- [Modulators](../modulators/index.md) for modulation components
