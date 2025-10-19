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

## See Also

- [Quick Example](../../index.md#quick-example) - Basic system construction with vortex phase mask
- [Typical Workflow](../index.md#typical-workflow-beam-splitter) - Complete example of building and optimizing an optical system
- [Fields](../fields/index.md) for ScalarField operations
- [Modes](../modes/index.md) for beam generation
- [OptimisersExt](../optimisers/index.md) for optimization
- [Metrics](../metrics/index.md) for loss functions
