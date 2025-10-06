# Sources

Optical field sources for system initialization.

## Overview

The `Sources` module provides:
- **Scalar field sources** for initializing optical systems
- **Trainable sources** for inverse design and optimization
- **Field generation** from templates or specifications

## Quick Example

```julia
using FluxOptics

# Create a Gaussian beam field
xv, yv = spatial_vectors(128, 128, 2.0, 2.0)
gaussian = Gaussian(20.0)
u0 = ScalarField(gaussian(xv, yv), (2.0, 2.0), 1.064)

# Static source (fixed beam profile)
source = ScalarSource(u0)
u = propagate(source)

# Trainable source for optimization
source_opt = ScalarSource(u0; trainable=true, buffered=true)
phase_mask = Phase(u0, (x, y) -> 0.01*(x^2 + y^2))
propagator = ASProp(u0, 500.0)
system = source_opt |> phase_mask |> propagator
```

## Key Types

- [`ScalarSource`](@ref): Source from scalar field template

## Key Functions

- [`get_source`](@ref get_source(::ScalarSource)): Access source field data
- [`propagate`](@ref): Generate field from source

## Design Patterns

### Static Sources
Use for fixed beam profiles:
```julia
source = ScalarSource(u0)  # trainable=false by default
```

### Trainable Sources
Enable optimization of source amplitude and phase:
```julia
source = ScalarSource(u0; trainable=true, buffered=true)
# ... optimization loop updates source.u0 ...
optimized_beam = get_source(source)
```

### In Optical Systems
Sources are typically the first component:
```julia
system = source |> component1 |> component2 |> propagator
```

## See Also

- [Fields](../../fields/index.md) for ScalarField operations
- [Modes](../../modes/index.md) for generating beam profiles
- [System](../system/index.md) for building optical systems

## Index

```@index
Modules = [FluxOptics.OpticalComponents]
Pages = ["sources.md"]
Order = [:type, :function]
```
