# Core

Foundational types and interfaces for optical components.

## Overview

The `Core` module defines the abstract component hierarchy, trainability system, and propagation interface used by all optical components in FluxOptics.

## Examples

### Component Trainability

```@example core
using FluxOptics

u = ScalarField(ones(ComplexF64, 128, 128), (2.0, 2.0), 1.064)

# Static component (not optimizable)
phase_static = Phase(u, (x, y) -> x^2)
istrainable(phase_static)  # false
```

```@example core
# Trainable component (optimizable parameters)
phase_train = Phase(u, (x, y) -> 0.0; trainable=true)

istrainable(phase_train)  # true
```

### Extracting Trainable Parameters

```@example core
# Access parameters for optimization
params = trainable(phase_train)

# params is a NamedTuple containing trainable arrays
keys(params)
```

### Propagation Interface

```@example core
# Forward propagation
u_fwd = propagate(u, phase_static, Forward)

# Backward propagation (applies conjugate phase)
u_bwd = propagate(u, phase_static, Backward)

# Forward and backward are complex conjugates
maximum(abs, u_fwd.electric - conj.(u_bwd.electric))
```

### Buffering Strategy

```@example core
# Buffered: pre-allocated gradient buffers (faster for iterative optimization)
phase_buffered = Phase(u, (x, y) -> 0.0; trainable=true, buffered=true)

# Unbuffered: allocates on-demand (lower memory footprint)
phase_unbuffered = Phase(u, (x, y) -> 0.0; trainable=true, buffered=false)

(isbuffered(phase_buffered), isbuffered(phase_unbuffered))
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

Where `M <: Trainability`:
- `Static`: Non-trainable, fixed parameters
- `Trainable{Buffered}`: Trainable with pre-allocated buffers
- `Trainable{Unbuffered}`: Trainable with on-demand allocation

## Key Concepts

### Trainability
Controls whether component parameters can be optimized:
- **Static**: Fixed parameters, no gradient computation, faster
- **Trainable**: Optimizable parameters, enables inverse design

### Buffering
Controls memory allocation strategy for trainable components:
- **Buffered**: Pre-allocates gradient and forward buffers → faster for repeated optimization
- **Unbuffered**: Allocates on-demand → lower memory usage, good for prototyping

### Bidirectional Propagation
All components support forward and backward propagation:
- **Forward**: Standard propagation through the component
- **Backward**: Computes the adjoint operation (required for gradient computation)
- Backward propagation is used internally during automatic differentiation
- Physical interpretation: time-reversal with conjugate fields

### Pure vs Custom Components

**Pure Components** (`AbstractPureComponent`):
- No manual gradient rules needed
- Zygote handles differentiation automatically
- Can wrap complex internal state or `AbstractCustomComponent`
- Examples: `ASPropZ`, `OpticalSequence`, `FourierWrapper`

**Custom Components** (`AbstractCustomComponent`):
- Custom gradient implementations
- More control over memory and computation
- Better performance for specific operations
- Examples: `Phase`, `Mask`, `ASProp`, `BPM`

### Source vs Pipe Components

**Sources** (`AbstractOpticalSource`):
- Generate fields
- Use `propagate(source)` (no input field argument)
- Placed at beginning of optical systems
- Example: `ScalarSource`

**Pipe Components** (`AbstractPipeComponent`):
- Transform existing fields
- Take input field as argument
- Use `propagate(u, component, direction)`
- Examples: `Phase`, `Mask`, `ASProp`, `FourierLens`

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

- [Typical Workflow](../../index.md#typical-workflow-beam-splitter) - Complete optimization example
- [Sources](../sources/index.md) - Field generation components
- [Modulators](../modulators/index.md) - Phase and amplitude modulation
- [System](../system/index.md) - Building optical systems

## Index

```@index
Modules = [FluxOptics.OpticalComponents]
Pages = ["core.md"]
Order = [:type, :function]
```
