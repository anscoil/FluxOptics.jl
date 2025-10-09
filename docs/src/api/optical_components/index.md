# Optical Components

Building blocks for optical systems and propagation.

## Overview

The `Optical Components` module provides all the building blocks needed to construct, propagate, and optimize optical systems. From abstract component interfaces to concrete propagators, modulators, and system composition tools.

## Module Organization

### [Core](core/index.md) - Foundation
Abstract types, trainability system, and propagation interface.
- Component hierarchy (Sources vs Pipes, Pure vs Custom)
- Trainability (Static, Trainable, Buffered)
- Bidirectional propagation (Forward/Backward)

### [Sources](sources/index.md) - Field Generation
Components that generate optical fields.
- `ScalarSource`: Field generation with optional trainability

### [Modulators](modulators/index.md) - Phase and Amplitude
Components that modify field amplitude and phase.
- `Phase`: Pure phase modulation
- `Mask`: Amplitude/complex transmission
- `TeaDOE`, `TeaReflector`: Diffractive optical elements

### [Fourier](fourier/index.md) - Frequency Domain
Operations in Fourier space.
- `FourierWrapper`: Apply components in frequency domain
- `FourierPhase`, `FourierMask`: Frequency-domain modulation
- `FourierOperator`: FFT/IFFT transformations

### [Free-Space Propagators](freespace/index.md) - Homogeneous Media
Field propagation through uniform media.
- `ASProp`, `ASPropZ`: Angular Spectrum method
- `RSProp`: Rayleigh-Sommerfeld diffraction
- `CollinsProp`, `FourierLens`: ABCD systems and Fourier optics
- `ParaxialProp`: Paraxial propagation
- `ShiftProp`: Geometric shift (ray optics)

### [Bulk Propagators](bulk/index.md) - Inhomogeneous Media
Beam Propagation Method for varying refractive index.
- `AS_BPM`: Split-step with Angular Spectrum
- `Shift_BPM`: Split-step with geometric shifts

### [Active Media](active/index.md) - Gain and Amplification
Components with optical gain.
- `GainSheet`: Saturable gain medium

### [Utilities](utilities/index.md) - Helper Components
Specialized tools for field manipulation and optimization.
- `PadCropOperator`: Memory-efficient propagation
- `TiltAnchor`: Off-axis beam tracking
- `BasisProjectionWrapper`: Reduced-parameter optimization

### [System](system/index.md) - Composition
Tools for building complete optical systems.
- `OpticalSystem`: Complete system with source
- `OpticalSequence`: Component sequence
- `FieldProbe`: Intermediate field capture
- Pipe operator `|>` for intuitive construction

## Quick Start

```julia
using FluxOptics

# 1. Create field and mode
u = ScalarField(ones(ComplexF64, 256, 256), (2.0, 2.0), 1.064)
xv, yv = spatial_vectors(256, 256, 2.0, 2.0)
gaussian = Gaussian(20.0)

# 2. Build system with pipe operator
source = ScalarSource(ScalarField(gaussian(xv, yv), (2.0, 2.0), 1.064); trainable=true)
phase = Phase(u, (x, y) -> 0.0; trainable=true)
lens = FourierLens(u, (1.0, 1.0), 1000.0)
prop = ASProp(u, 1000.0)

system = source |> phase |> lens |> prop

# 3. Execute
result = system()
output = result.out

# 4. Optimize
target = ScalarField(target_data, (2.0, 2.0), 1.064)
metric = PowerCoupling(target)
loss() = 1.0 - metric(system())[]
```

## Component Types

**Sources** generate fields from nothing:
- Entry point of optical systems
- Use `propagate(source)` (no input field)

**Pipe Components** transform existing fields:
- Chain with `|>` operator
- Use `propagate(u, component, direction)`

**Pure Components** have functional interface:
- Zygote-compatible automatic differentiation
- No custom gradient rules needed

**Custom Components** maintain explicit state:
- Custom gradient implementations
- Fine-grained memory control

## Design Patterns

### Static vs Trainable

```julia
# Static: fixed parameters
phase_static = Phase(u, (x, y) -> x^2)

# Trainable: optimizable
phase_opt = Phase(u, (x, y) -> 0.0; trainable=true, buffered=true)
```

### System Composition

```julia
# Pipe operator chains components
system = source |> comp1 |> comp2 |> comp3

# Options passed as NamedTuple
system = system |> (; merge_components=true, inplace=false)
```

### Bidirectional Propagation

```julia
# Forward propagation
u_fwd = propagate(u, component, Forward)

# Backward propagation (e.g., adjoint method)
u_bwd = propagate(u, component, Backward)
```

## See Also

- [Fields](../fields/index.md) for ScalarField operations
- [Modes](../modes/index.md) for beam generation
- [OptimisersExt](../optimisers/index.md) for optimization
- [Metrics](../metrics/index.md) for loss functions
