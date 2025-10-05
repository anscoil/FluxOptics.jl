# API Reference

This section provides detailed documentation for all FluxOptics.jl modules and functions.

## Quick Navigation

FluxOptics is organized into focused modules for different aspects of optical simulation and inverse design:

### Foundation

**[GridUtils](gridutils/index.md)** - Coordinate systems and transformations
- Spatial coordinate generation for optical grids
- 2D transformations (translations, rotations)
- Coordinate composition for complex geometries

**[Modes](modes/index.md)** - Optical mode generation
- Gaussian beam families (Gaussian, Hermite-Gaussian, Laguerre-Gaussian)
- Spatial layouts for multi-mode configurations
- Speckle generation with controlled statistics

**[Fields](fields/index.md)** - Field representation and operations
- `ScalarField` type for optical fields
- Multi-wavelength and tilt support
- Power, intensity, and field comparison operations

### Propagation & Components

**[Optical Components](optical_components/index.md)** - Building blocks for optical systems
- **Core**: Abstract types, sources, static components (Phase, Mask, TeaDOE)
- **Free-Space Propagators**: Angular Spectrum, Rayleigh-Sommerfeld, Collins integral
- **Bulk Propagators**: Beam Propagation Method for inhomogeneous media
- **Active Media**: Gain sheets and amplifiers
- Optical systems with piping syntax

### Optimization

**[OptimisersExt](optimisers/index.md)** - Optimization algorithms and rules
- Custom optimization rules (Descent, Momentum, FISTA)
- Proximal operators for constrained optimization
- Integration with Optimisers.jl ecosystem

**[Metrics](metrics/index.md)** - Loss functions for inverse design
- Field overlap metrics (DotProduct, PowerCoupling)
- Field and intensity matching objectives
- Custom gradient implementations for efficiency

## Design Philosophy

FluxOptics follows these principles:

- **Differentiable by design**: All components work with automatic differentiation
- **GPU-ready**: Seamless CUDA.jl integration for acceleration
- **Composable**: Build complex systems from simple components
- **Efficient**: Pre-allocated buffers on-demand and optimized kernels
- **Flexible**: Support for multi-wavelength, multi-mode, and off-axis propagation

## Typical Workflow

```julia
using FluxOptics

# 1. Define spatial grid and modes
xv, yv = spatial_vectors(128, 128, 2.0, 2.0)
gaussian = Gaussian(20.0)

# 2. Create field
u = ScalarField(gaussian(xv, yv), (2.0, 2.0), 1.064)

# 3. Build optical system
source = ScalarSource(u; trainable=true)
phase = Phase(u, (x, y) -> 0.0; trainable=true)
propagator = ASProp(u, 1000.0)
system = source |> phase |> propagator

# 4. Define optimization objective
target = ScalarField(target_data, (2.0, 2.0), 1.064)
metric = PowerCoupling(target)

# 5. Optimize
using Zygote
loss() = 1.0 - metric(system())[]
# ... gradient descent ...
```

## Module Overview

```@contents
Pages = [
    "gridutils/index.md",
    "modes/index.md",
    "fields/index.md",
    "optical_components/index.md",
    "optimisers/index.md",
    "metrics/index.md",
]
Depth = 1
```

