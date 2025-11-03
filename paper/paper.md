---
title: 'FluxOptics.jl: A Differentiable Wave Optics Framework for Inverse Design in Julia'
tags:
  - Julia
  - optics
  - inverse design
  - automatic differentiation
  - beam propagation
  - diffractive optics
  - computational optics
authors:
  - name: Nicolas Barré
    orcid: 0000-0002-4460-4151
    affiliation: 1
affiliations:
 - name: Independent Researcher, France
   index: 1
date: 3 November 2025
bibliography: paper.bib
---

# Summary

FluxOptics.jl is a Julia package for simulating optical field propagation with full support for automatic differentiation. It enables gradient-based inverse design of optical components through efficient, differentiable wave propagation.

The package currently implements scalar field propagation methods (computationally efficient alternatives to finite-difference time-domain simulations), making it particularly suited for designing optical elements compatible with emerging additive manufacturing techniques such as direct laser writing (DLW) and two-photon polymerization (TPP), as well as low-cost characterization methods such as intensity-only diffraction tomography.

FluxOptics.jl provides multiple propagation methods including Angular Spectrum, Rayleigh-Sommerfeld, and Beam Propagation Method, along with a composable architecture for building complex optical systems, GPU acceleration, and advanced optimization tools including proximal operators for constrained inverse design. The extensible architecture is designed to support future vector field propagation with polarization-dependent components and dielectric metasurfaces.

# Statement of Need

Inverse design of optical components (determining the structure of an optical element to achieve desired functionality) has become increasingly important with the rise of freeform optics [@Schmidt2020; @Barre2025], diffractive optical elements (DOEs) [@Dinc2020], and metasurfaces [@Molesky2018; @Peurifoy2018]. Traditional design approaches rely on iterative forward simulation and manual parameter tuning, which becomes intractable for high-dimensional design spaces. Gradient-based optimization using automatic differentiation (AD) has emerged as a powerful alternative, enabling efficient exploration of complex design spaces [@Hughes2018; @Minkov2020].

However, existing tools face several limitations. Full-wave electromagnetic solvers like FDTD provide high accuracy but are computationally prohibitive for optimization, often requiring hours per forward simulation and limited to 2D or small 3D domains [@Oskooi2010]. Python packages like TorchOptics [@TorchOptics] provide differentiable scalar wave propagation but suffer from performance bottlenecks and limited composability. Moreover, most existing tools are not designed for the emerging paradigm of frugal optics: fast, lightweight propagation methods compatible with additive manufacturing and low-cost characterization techniques.

FluxOptics.jl addresses these gaps through several key innovations. First, it provides high-performance differentiable propagation where all components are fully differentiable with automatic differentiation support via Zygote.jl [@Innes2019] and Enzyme.jl [@Moses2021], enabling gradient-based inverse design. Implemented in Julia [@Bezanson2017], the package leverages just-in-time compilation and multiple dispatch to achieve high-performance CPU and GPU implementations that significantly outperform existing Python-based tools.

Second, the package offers a flexible component architecture with two design patterns for implementing optical components. Components inheriting from `AbstractPureComponent` implement a pure `propagate` method that returns a transformed field without side effects. Zygote automatically derives the adjoint for gradient backpropagation via automatic differentiation, enabling rapid prototyping. Components inheriting from `AbstractCustomComponent` implement the full interface including in-place forward propagation, adjoint propagation with gradient computation, and buffer management for trainable parameters. This manual implementation provides maximum performance and memory control without requiring users to write low-level ChainRules. This dual approach balances ease of prototyping with production-level optimization.

Third, FluxOptics.jl emphasizes composability. Optical systems are built using Julia's pipe operator (`|>`), allowing intuitive construction of complex cascaded systems. Non-trainable adjacent components with compatible types can be merged to improve efficiency (e.g., adjacent phase masks). The `FieldProbe` mechanism enables capturing intermediate field states for multi-objective optimization, visualization, and debugging.

Fourth, the package focuses on frugal optics by implementing computationally efficient propagation methods (Angular Spectrum, Rayleigh-Sommerfeld, paraxial and non-paraxial BPM). This targets applications in additive manufacturing of optical elements and low-cost characterization techniques. The package already demonstrates intensity-based waveguide tomography, a key application for accessible optical metrology.

Finally, the architecture is designed for extensibility from scalar to vector field propagation, with planned support for polarization-dependent components such as dielectric metasurfaces with nanopillars.

FluxOptics.jl emerged from practical research challenges encountered across diverse optical applications, from laser cavity design [@Barre2014] to waveguide characterization [@Barre2021] and multimode beam control [@Barre2022OL; @Barre2022CIRP]. Rather than developing specialized tools for each problem, the package provides a unified framework that addresses the common computational patterns underlying these applications. The library has already been used for cavity eigenmode analysis, phase retrieval, multi-wavelength beam shaping, waveguide tomography, and multimode intensity shaping, demonstrating its versatility. By consolidating these approaches into a single, well-tested package with consistent API design, FluxOptics.jl aims to accelerate research in inverse optical design and make gradient-based optimization accessible to a broader community.

# Key Features

## Component Architecture and Automatic Differentiation

All optical components in FluxOptics.jl inherit from a unified abstract type hierarchy that enables automatic differentiation while providing flexibility in implementation strategy. The root type `AbstractOpticalComponent{M}` branches into two categories based on functionality:

**AbstractOpticalSource**: Components that generate optical fields (e.g., laser sources, input beams). Sources are accessed via `propagate(source)` without an input field argument, producing the initial field for an optical system.

**AbstractPipeComponent**: Components that transform existing optical fields as they propagate through the system (e.g., lenses, phase masks, propagators). These components are called via `propagate(u, component, direction)`.

Connecting components with the pipe operator (`|>`) creates an `OpticalSystem`, which is a callable object that executes the complete optical simulation when invoked.

Within each category, components can follow one of two implementation patterns:

**Pure components** (`AbstractPureComponent`, `AbstractPureSource`): Require only a pure `propagate` method implementation. Zygote (or Enzyme) automatically derives the adjoint for gradient backpropagation via automatic differentiation. This minimal interface enables rapid prototyping.

**Custom components** (`AbstractCustomComponent`, `AbstractCustomSource`): Require implementing the full interface including in-place forward propagation, adjoint propagation with gradient computation, and buffer management for trainable parameters. This manual implementation provides fine-grained control over memory allocation and computational efficiency.

Both component types can be mixed freely within a single `OpticalSystem`, allowing developers to prototype quickly with pure components and optimize critical paths with custom implementations as needed.

The type parameter `M <: Trainability` controls optimization behavior: `Static` for fixed components, `Trainable{Buffered}` for components with pre-allocated gradient buffers (faster iteration), and `Trainable{Unbuffered}` for memory-efficient on-demand allocation.

The package provides advanced optimization tools built on Optimisers.jl:
- Proximal operators for constrained optimization (TV regularization, sparsity via ISTA, box constraints)
- Per-component learning rates via `make_rules`
- FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) [@Beck2009] acceleration for faster convergence
- Flexible memory management through buffering strategies

## Propagation Methods

FluxOptics.jl provides a comprehensive set of scalar wave propagation methods:

- **Free-space propagation**: Angular Spectrum Method (AS), Rayleigh-Sommerfeld diffraction, Collins integral for ABCD systems. Propagators natively support tilted beam propagation by storing tilt information in the field representation. Optional tilt tracking keeps the beam centered in the computational window during propagation through cascaded systems.
- **Graded-index media**: Beam Propagation Method (BPM) with support for spatially-varying refractive index profiles. Includes both paraxial and non-paraxial formulations for tilted beam propagation through inhomogeneous media.
- **Fourier optics**: Fourier lenses, frequency-domain filtering
- **Active media**: Stationary gain sheets with saturable amplification for laser cavity simulation

All propagators support bidirectional propagation by simply changing the sign of the propagation distance (±z), enabling simulation of optical systems in both forward and backward directions. During optimization, the adjoint propagation required for gradient computation is handled automatically by the automatic differentiation framework.

## Optical Components

The package provides building blocks for complex optical systems. Sources can be `Trainable` or `Static` scalar field sources. Modulators include phase masks, amplitude masks, and Thin Element Approximation (TEA) diffractive elements. System composition uses pipe syntax (`source |> component1 |> component2 |> ...`) for intuitive construction. `FieldProbe` objects capture intermediate field states, enabling construction of loss functions that depend on fields at multiple planes, as well as visualization and debugging. Adjacent non-trainable components with compatible types can be merged for efficiency when enabled via `merge_components=true`.

## Multi-Wavelength and Multimode Support

FluxOptics.jl natively supports polychromatic propagation for achromatic or chromatic optical element design, and multimode fields for mode coupling and multimode fiber applications.

## GPU Acceleration

Seamless GPU acceleration via CUDA.jl:
```julia
using CUDA
u = cu(ScalarField(...))  # Move to GPU
source = ScalarSource(u)   # Source inherits GPU context from u
doe = Phase(u, ...)        # Components inherit GPU context from u
system = source |> doe |> propagator  # Automatic GPU execution
```

All operations transparently run on GPU with minimal code changes.

## Performance Benchmark

Comparison with TorchOptics [@TorchOptics] on a [beam splitter inverse design task](https://anscoil.github.io/FluxOptics.jl/stable/api/#Typical-Workflow:-Beam-Splitter) (250×250 grid, 3 DOEs, 200 optimization iterations):

| Platform | TorchOptics | FluxOptics.jl | Speedup |
|----------|-------------|---------------|---------|
| CPU (multi-threaded) | ~7s | ~5s | 1.4× |
| GPU (NVIDIA RTX 4070 Super) | ~3.5s | ~0.27s | 13× |

CPU memory footprint: 41 MiB total allocation for 200 iterations (~205 KiB per iteration).

Beyond raw speed, FluxOptics.jl demonstrates excellent memory scalability with a footprint of only 41 MiB for 200 optimization iterations. This enables scaling to large multimode problems with hundreds of modes across multiple propagation planes, critical for applications in multimode fibers, spatial mode multiplexing, and complex cascaded optical systems.

# Tutorials and Documentation

FluxOptics.jl provides comprehensive documentation including five detailed tutorials:

1. **Fox-Li Cavity Simulation**: Finding laser cavity eigenmodes in semi-degenerate resonators, demonstrating stationary gain media and iterative propagation
2. **Field Retrieval from Intensity**: Reconstructing complex optical fields (amplitude and phase) from intensity-only measurements via gradient-based optimization, generalizing classical iterative projection methods [@Fienup1982] through automatic differentiation
3. **Multi-Wavelength Beam Shaping**: Designing chromatic DOE cascades for independent control of red, green, and blue beams
4. **Waveguide Tomography**: Reconstructing refractive index profiles from angle-resolved intensity data, demonstrating intensity-based characterization
5. **Multimode Intensity Shaping**: Shaping 105 Laguerre-Gaussian modes using only 2 cascaded DOEs, demonstrating TV-norm regularization and basis projection for cylindrical symmetry enforcement

Complete API documentation and interactive examples are available at [https://anscoil.github.io/FluxOptics.jl/stable/](https://anscoil.github.io/FluxOptics.jl/stable/).

# Acknowledgments

There are no acknowledgments to declare and no conflicts of interest to disclose.

# References
