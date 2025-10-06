# API Reference

Complete documentation for all FluxOptics.jl modules and functions.

## Quick Navigation

FluxOptics is organized into focused modules for different aspects of optical simulation and inverse design:

### Foundation

**[GridUtils](gridutils/index.md)** - Coordinate systems and transformations
- Spatial coordinate generation for optical field grids
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

### Optical Components

**[Optical Components](optical_components/index.md)** - Building blocks for optical systems

Core architecture:
- **[Core](optical_components/core/index.md)** - Abstract types, trainability system, propagation interface
- **[Sources](optical_components/sources/index.md)** - Field generation (ScalarSource)
- **[Modulators](optical_components/modulators/index.md)** - Phase and amplitude modulation (Phase, Mask, TeaDOE)
- **[Fourier](optical_components/fourier/index.md)** - Frequency-domain operations (FourierWrapper, FourierPhase, FourierMask)
- **[System](optical_components/system/index.md)** - System composition (OpticalSystem, OpticalSequence, FieldProbe)
- **[Utilities](optical_components/utilities/index.md)** - Helper components (PadCropOperator, TiltAnchor, BasisProjectionWrapper)

Propagation methods:
- **[Free-Space Propagators](optical_components/freespace/index.md)** - Angular Spectrum, Rayleigh-Sommerfeld, Collins integral
- **[Bulk Propagators](optical_components/bulk/index.md)** - Beam Propagation Method for inhomogeneous media

Active components:
- **[Active Media](optical_components/active/index.md)** - Gain sheets and amplifiers

### Optimization

**[OptimisersExt](optimisers/index.md)** - Optimization algorithms and rules
- Custom optimization rules (Descent, Momentum, FISTA)
- Proximal operators for constrained optimization
- Per-component rule assignment
- Integration with Optimisers.jl ecosystem

**[Metrics](metrics/index.md)** - Loss functions for inverse design
- Field overlap metrics (DotProduct, PowerCoupling)
- Field and intensity matching objectives
- Custom gradient implementations for efficiency

## Design Philosophy

FluxOptics follows these principles:

- **Differentiable by design**: All components work with automatic differentiation
- **GPU-ready**: Seamless CUDA.jl integration for acceleration
- **Composable**: Build complex systems from simple components using pipe operator
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
params = trainable(system)
# ... gradient descent loop ...
```

## Module Overview

### Foundation Modules

These modules provide the basic building blocks for optical field representation and manipulation.

**GridUtils** creates coordinate systems for evaluating optical modes and components on spatial grids.

**Modes** generates common optical beam profiles (Gaussian, Hermite-Gaussian, Laguerre-Gaussian) and spatial layouts for multi-mode configurations.

**Fields** provides the `ScalarField` type, the central data structure representing optical fields with associated grid information, wavelength, and propagation direction.

### Optical Components

The heart of FluxOptics, providing all optical elements and system composition tools.

**Core** defines the abstract component hierarchy, trainability system (Static/Trainable/Buffered), and bidirectional propagation interface used by all components.

**Sources** generate initial optical fields (e.g., `ScalarSource`). Sources are trainable, allowing optimization of the input beam profile.

**Modulators** modify field amplitude and phase: `Phase` for pure phase modulation, `Mask` for amplitude/complex transmission, `TeaDOE` for diffractive elements with a physical thickness that satisfies the Thin Element Approximation.

**Fourier** provides frequency-domain operations. `FourierWrapper` applies components in Fourier space, while `FourierPhase` and `FourierMask` are convenient constructors for frequency-domain filtering.

**System** enables optical system construction. Use the pipe operator `|>` to chain components: `source |> phase |> lens |> propagator`. Systems are callable and fully differentiable. `FieldProbe` captures intermediate fields for use in custom objective functions, visualizations, and debugging.

**Utilities** contains helper components: `PadCropOperator` for memory-efficient propagation, `TiltAnchor` for off-axis beam tracking, `BasisProjectionWrapper` for reduced-parameter optimization.

**Free-Space Propagators** implement field propagation through homogeneous media: Angular Spectrum method (ASProp), Rayleigh-Sommerfeld diffraction (RSProp), Collins integral for ABCD systems, Fourier lenses.

**Bulk Propagators** use Beam Propagation Method (BPM) for inhomogeneous media with spatially-varying refractive index. Supports paraxial  and non-paraxial tilted propagation.

**Active Media** models gain and amplification with saturable gain sheets.

### Optimization Modules

**OptimisersExt** provides optimization algorithms and proximal operators. Use `make_rules` for per-component learning rates, `ProxRule` for constrained optimization with regularization.

**Metrics** defines loss functions for inverse design: `PowerCoupling` for mode matching, `SquaredFieldDifference` for field shaping, `SquaredIntensityDifference` for intensity targets.
