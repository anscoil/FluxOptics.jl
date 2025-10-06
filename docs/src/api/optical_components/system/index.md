# System

Optical system composition and execution.

## Overview

The `System` module provides:
- **Optical systems** combining sources and components
- **Component sequences** for reusable propagation chains
- **Pipe operator syntax** for intuitive system building
- **Component merging** for performance optimization
- **Field probes** for intermediate field capture

## Quick Example

```julia
using FluxOptics

u = ScalarField(ones(ComplexF64, 256, 256), (2.0, 2.0), 1.064)

# Build system with pipe operator
source = ScalarSource(u; trainable=true)
phase = Phase(u, (x, y) -> 0.0; trainable=true)
lens = FourierLens(u, (2.0, 2.0), 1000.0)
prop = ASProp(u, 500.0)

system = source |> phase |> lens |> prop

# Execute system
result = system()
output_field = result.out

# With component merging (more efficient)
phase1 = Phase(u, (x, y) -> x^2)
phase2 = Phase(u, (x, y) -> y^2)
system = source |> phase1 |> phase2 |> (; merge_components=true)
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
