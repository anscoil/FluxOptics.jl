# Sources

Optical field sources for system initialization.

## Overview

The `Sources` module provides components that generate optical fields to initialize optical systems. Sources can be static (fixed beam profile) or trainable (optimizable during inverse design).

## Quick Example

### Creating Sources

```@example sources
using FluxOptics

# Create spatial grid and mode
xv, yv = spatial_vectors(128, 128, 2.0, 2.0)
gaussian = Gaussian(20.0)
u0 = ScalarField(gaussian(xv, yv), (2.0, 2.0), 1.064)

# Static source (fixed beam profile)
source_static = ScalarSource(u0)

istrainable(source_static)  # false
```

```@example sources
# Trainable source (optimizable)
source_train = ScalarSource(u0; trainable=true, buffered=true)

(istrainable(source_train), isbuffered(source_train))
```

### Using Sources in Systems

```@example sources
# Generate field from source
u_generated = propagate(source_static)

size(u_generated)  # Same as template
```

```@example sources
# Build system with trainable source
phase_mask = Phase(u0, (x, y) -> 0.01*(x^2 + y^2))
propagator = ASProp(u0, 500.0)

system = source_train |> phase_mask |> propagator

# Execute system
result = system()
output = result.out

power(output)[]  # Propagated through system
```

### Accessing Source Data

```@example sources
# Extract current source field
current_field = get_source(source_train)

# Access trainable parameters (for optimization)
params = trainable(source_train)

keys(params)
```

## Key Types

- [`ScalarSource`](@ref): Source from scalar field template

## Key Functions

- [`get_source`](@ref get_source(::ScalarSource)): Access source field data
- [`propagate`](@ref): Generate field from source

## Usage Patterns

### Static Sources
Use for fixed, known beam profiles:
```julia
source = ScalarSource(u0)  # trainable=false by default
```

### Trainable Sources  
Enable optimization of source amplitude and phase during inverse design:
```julia
source = ScalarSource(u0; trainable=true, buffered=true)
# ... optimization updates source field ...
optimized_beam = get_source(source)
```

### Multi-Mode Sources
Sources support multi-mode fields:
```julia
# Create multi-mode field
modes = hermite_gaussian_groups(15.0, 3)
mode_data = stack([m(xv, yv) for m in modes], dims=3)
u_multi = ScalarField(mode_data, (2.0, 2.0), 1.064)

source_multi = ScalarSource(u_multi; trainable=true)
```

## See Also

- [Typical Workflow](../../index.md#typical-workflow-beam-splitter) - Complete optimization example
- [Modes](../../modes/index.md) - Generating beam profiles
- [Fields](../../fields/index.md) - ScalarField operations
- [System](../system/index.md) - Building optical systems

## Index

```@index
Modules = [FluxOptics.OpticalComponents]
Pages = ["sources.md"]
Order = [:type, :function]
```
