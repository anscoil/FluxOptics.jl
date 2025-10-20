# System

Optical system composition and execution.

## Overview

The `System` module provides:
- **Optical systems** combining sources and components
- **Component sequences** for reusable propagation chains
- **Pipe operator syntax** for intuitive system building
- **Component merging** for performance optimization
- **Field probes** for intermediate field capture

## Examples

### Basic System Construction

```@example system1
using FluxOptics

# Create field and components
xv, yv = spatial_vectors(256, 256, 2.0, 2.0)
u = ScalarField(Gaussian(20.0)(xv, yv), (2.0, 2.0), 1.064)

source = ScalarSource(u)
phasemask = Phase(u, (x, y) -> 0.01*(x^2 + y^2))
prop = ASProp(u, 500.0)

# Build system with pipe operator
system = source |> phasemask |> prop

# Execute system
result = system()
power(result.out)[]
```

### Component Merging

```@example system2
using FluxOptics

xv, yv = spatial_vectors(128, 128, 2.0, 2.0)
u = ScalarField(Gaussian(15.0)(xv, yv), (2.0, 2.0), 1.064)

source = ScalarSource(u)

# Create multiple phase masks
phase1 = Phase(u, (x, y) -> 0.01*x^2)
phase2 = Phase(u, (x, y) -> 0.01*y^2)
phase3 = Phase(u, (x, y) -> 0.01*x*y)

# Without merging: 3 separate phase operations
system_no_merge = source |> phase1 |> phase2 |> phase3
components_no_merge = get_components(system_no_merge)

# With merging: phases combined into single operation
system_merged = source |> phase1 |> phase2 |> phase3 |> (; merge_components=true)
components_merged = get_components(system_merged)

# Compare number of operations
(length(components_no_merge), length(components_merged))
```

### Field Probes

```@example system3
using FluxOptics, CairoMakie

xv, yv = spatial_vectors(256, 256, 1.0, 1.0)
u = ScalarField(Gaussian(30.0)(xv, yv), (1.0, 1.0), 1.064)

source = ScalarSource(u)
phasemask = Phase(u, (x, y) -> 0.01*(x^2 + y^2))
lens = FourierLens(u, (2.0, 2.0), 1000.0)
prop = ASProp(u, (2.0, 2.0), 4000.0)

# Insert probes to capture intermediate fields
probe1 = FieldProbe()
probe2 = FieldProbe()

system = source |> phasemask |> probe1 |> lens |> probe2 |> prop

# Execute and access intermediate fields
result = system()

# Access fields at probe locations
field_after_phase = result.probes[probe1]
field_after_lens = result.probes[probe2]
final_field = result.out

# Show intensity at each stage
visualize((field_after_phase, field_after_lens, final_field), (intensity, phase);
    colormap=(:inferno, :viridis), height=120)
```

## Key Types

- [`OpticalSystem`](@ref): Complete optical system with source
- [`AbstractSequence`](@ref): Abstract type for defining optical sequences
- [`OpticalSequence`](@ref): Component sequence without source
- [`FieldProbe`](@ref): Intermediate field capture

## Key Functions

- [`get_source`](@ref get_source(::OpticalSystem)): Extract source from system
- [`get_components`](@ref): Extract component sequence
- [`get_sequence`](@ref): Extract components from sequence

## Design Patterns

### System Construction
Use pipe operator for readable system building:
```julia
system = source |> component1 |> component2 |> component3
```

### Component Merging
Enable merging for performance:
```julia
system = source |> phase1 |> phase2 |> (; merge_components=true)
# Adjacent compatible components are merged
```

### Field Probes
Insert probes to capture intermediate fields:
```julia
probe = FieldProbe()
system = source |> phase |> probe |> lens
result = system()
intermediate = result.probes[probe]
```

## See Also

- [Core](../core/index.md) for component interface
- [Sources](../sources/index.md) for field generation
- [Modulators](../modulators/index.md) for components

## Index

```@index
Modules = [FluxOptics.OpticalComponents]
Pages = ["system.md"]
Order = [:type, :function]
```
