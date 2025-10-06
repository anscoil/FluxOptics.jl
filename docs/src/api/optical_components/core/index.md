# Core

Foundational types and interfaces for optical components.

## Overview

The `Core` module provides:
- **Abstract component hierarchy** for all optical elements
- **Trainability system** for optimization and inverse design
- **Propagation interface** for field transformations
- **Direction handling** for forward and backward propagation
- **Common utilities** for component management

## Quick Example

```julia
using FluxOptics

# Create field
u = ScalarField(ones(ComplexF64, 128, 128), (2.0, 2.0), 1.064)

# Static component (no optimization)
phase_static = Phase(u, (x, y) -> x^2)
istrainable(phase_static)  # false

# Trainable component (optimizable)
phase_train = Phase(u, (x, y) -> 0.0; trainable=true, buffered=true)
istrainable(phase_train)  # true
isbuffered(phase_train)   # true

# Propagation
u_out = propagate(u, phase_train, Forward)

# Access trainable parameters
params = trainable(phase_train)  # (; ϕ = ...)
```

## Component Hierarchy

```
AbstractOpticalComponent{M}
├── AbstractOpticalSource{M}        # Generates fields
│   ├── AbstractPureSource{M}       # Functional sources
│   └── AbstractCustomSource{M}     # Stateful sources
└── AbstractPipeComponent{M}        # Transforms fields
    ├── AbstractPureComponent{M}    # Functional components
    └── AbstractCustomComponent{M}  # Stateful components
```

## Key Concepts

### Trainability
Components can be `Static` (fixed parameters) or `Trainable` (optimizable):
```julia
# Static: no gradients, faster
component_static = Phase(u, (x, y) -> x^2)

# Trainable: enables optimization
component_train = Phase(u, (x, y) -> 0.0; trainable=true)
```

### Buffering
Trainable components can pre-allocate buffers for efficiency:
```julia
# Unbuffered: allocates on-the-fly
component = Phase(u, (x, y) -> 0.0; trainable=true, buffered=false)

# Buffered: pre-allocated (faster for iterative optimization)
component = Phase(u, (x, y) -> 0.0; trainable=true, buffered=true)
```

### Propagation Direction
All components support bidirectional propagation:
```julia
# Forward propagation (default)
u_fwd = propagate(u, component, Forward)

# Backward propagation
u_bwd = propagate(u, component, Backward)
```

## Key Types

**Abstract Types**
- [`AbstractOpticalComponent`](@ref): Root type for all components
- [`AbstractOpticalSource`](@ref): Field generators
- [`AbstractPipeComponent`](@ref): Field transformers
- [`AbstractPureComponent`](@ref): Functional interface components
- [`AbstractCustomComponent`](@ref): Stateful components

**Trainability**
- [`Trainability`](@ref): Abstract trainability type
- [`Static`](@ref): Non-trainable components
- [`Trainable`](@ref): Trainable components
- [`Buffering`](@ref): Buffer management strategy
- [`Buffered`](@ref), [`Unbuffered`](@ref): Buffering modes

**Direction**
- [`Direction`](@ref): Abstract direction type
- [`Forward`](@ref), [`Backward`](@ref): Propagation directions

## Key Functions

**Component Interface**
- [`propagate`](@ref): Apply component (creates copy)
- [`propagate!`](@ref): Apply component in-place
- [`get_data`](@ref): Access component parameters
- [`trainable`](@ref trainable(::AbstractOpticalComponent{Static})): Extract trainable parameters

**Component Queries**
- [`istrainable`](@ref): Check if component is trainable
- [`isbuffered`](@ref): Check if component uses buffers

## See Also

- [Sources](../sources/index.md) for field generation
- [Modulators](../modulators/index.md) for phase and amplitude modulation
- [System](../system/index.md) for building optical systems
- [Fields](../../fields/index.md) for ScalarField operations

## Index

```@index
Modules = [FluxOptics.OpticalComponents]
Pages = ["core.md"]
Order = [:type, :function]
```
