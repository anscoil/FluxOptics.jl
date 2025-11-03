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

FluxOptics.jl is a Julia package for simulating optical field propagation with full support for automatic differentiation. It enables gradient-based inverse design of optical components, which consists in determining the structure of an optical element (lens, diffraction grating, phase mask) that produces a desired light pattern or functionality. The package implements scalar wave propagation methods that are computationally efficient alternatives to finite-difference time-domain (FDTD) simulations, making it particularly suited for designing optical elements compatible with additive manufacturing techniques such as direct laser writing and two-photon polymerization, as well as low-cost characterization methods like intensity-only diffraction tomography.

FluxOptics.jl provides multiple propagation algorithms for free-space and graded-index media, a composable architecture for building complex optical systems, GPU acceleration, and optimization tools including proximal operators for constrained inverse design. The architecture supports current scalar field applications and is designed to extend to vector field propagation for polarization-dependent components and dielectric metasurfaces.

# Statement of Need

Inverse design of optical components has become increasingly important with the rise of freeform optics [@Schmidt2020; @Barre2025], diffractive optical elements (DOEs) [@Dinc2020], and metasurfaces [@Molesky2018; @Peurifoy2018]. Traditional optimization approaches include gradient-free methods (evolutionary algorithms, Bayesian optimization, stochastic search) which can be effective for low-dimensional problems with up to hundreds of parameters. However, these methods become intractable when the design space scales to thousands or millions of parameters, as is typical for spatially-resolved optical elements defined on computational grids. Gradient-based optimization using automatic differentiation has emerged as the solution for such high-dimensional design spaces [@Hughes2018; @Minkov2020], enabling efficient convergence by exploiting gradient information at computational cost comparable to a single forward simulation.

However, existing tools face several limitations. Full-wave electromagnetic solvers like FDTD provide high accuracy but are computationally prohibitive for optimization, often requiring hours per forward simulation and limited to 2D or small 3D domains [@Oskooi2010]. Python packages like TorchOptics [@TorchOptics] provide differentiable scalar wave propagation but suffer from performance bottlenecks.

FluxOptics.jl addresses these gaps through several key innovations. First, it provides high-performance differentiable propagation via Zygote.jl [@Innes2019] or Enzyme.jl [@Moses2021]. Implemented in Julia [@Bezanson2017], the package achieves CPU and GPU implementations that significantly outperform existing Python-based tools.

Second, the package provides an extensible component architecture designed for users to implement their own optical components. The interface offers two implementation patterns depending on performance requirements: rapid prototyping with automatic adjoint derivation via automatic differentiation, or fine-grained control over memory allocations and gradient computation for production-level performance.

Third, FluxOptics.jl emphasizes composability. Optical systems are built using Julia's pipe operator (`|>`), allowing intuitive construction of cascaded systems. The `FieldProbe` mechanism enables capturing intermediate field states for multi-objective optimization, visualization, and debugging.

Fourth, the package implements efficient propagation methods through kernel caching strategies that avoid redundant computations. This makes scalar wave propagation practical for iterative optimization in applications such as additive manufacturing and intensity-based waveguide tomography.

Finally, the architecture is designed for extensibility from scalar to vector field propagation, with planned support for polarization-dependent components.

FluxOptics.jl emerged from practical research challenges encountered across diverse optical applications, from laser cavity design [@Barre2014] to waveguide characterization [@Barre2021] and multimode beam control [@Barre2022OL; @Barre2022CIRP]. Rather than developing specialized tools for each problem, the package provides a unified framework that addresses the common computational patterns underlying these applications. By consolidating these approaches into a single, well-tested package with consistent API design, FluxOptics.jl aims to accelerate research in inverse optical design and make gradient-based optimization accessible to a broader community.

# Key Features

FluxOptics.jl provides a comprehensive set of tools for differentiable optical simulation and inverse design. The package combines efficient propagation algorithms with automatic differentiation support, enabling both rapid prototyping and production-level optimization of optical systems.

## Component Architecture and Automatic Differentiation

All optical components inherit from a unified abstract type hierarchy that enables automatic differentiation. Components are divided into sources (generating optical fields) and pipe components (transforming fields). Connecting components with the pipe operator (`|>`) creates an `OpticalSystem` that executes the complete optical simulation when invoked.

The architecture offers two implementation patterns: **Pure components** require only a pure `propagate` method where automatic differentiation (Zygote or Enzyme) automatically derives adjoints for gradient backpropagation, enabling rapid prototyping. **Custom components** implement the full interface with in-place propagation and manual adjoint specification, providing fine-grained control over memory allocation and computational efficiency. Both types can be mixed freely within a single system.

The package provides advanced optimization tools built on Optimisers.jl:

- Proximal operators for constrained optimization (TV regularization, sparsity via ISTA, box constraints)
- Per-component learning rates
- FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) [@Beck2009] acceleration for faster convergence

## Optical Components and Field Propagation

FluxOptics.jl provides a comprehensive set of optical components including sources, modulators, and propagation methods.

**Free-space propagation** includes the Angular Spectrum Method, Rayleigh-Sommerfeld diffraction, and Collins integral for ABCD systems. Propagators natively support tilted beam propagation by storing tilt information in the field representation. Optional tilt tracking keeps the beam centered in the computational window during propagation through cascaded systems.

**Graded-index media** are handled through the Beam Propagation Method with support for spatially-varying refractive index profiles. This includes both paraxial and non-paraxial formulations for tilted beam propagation through inhomogeneous media.

**Fourier optics** capabilities include Fourier lenses and frequency-domain filtering.

**Active media** can be simulated using stationary gain sheets with saturable amplification for laser cavity simulation.

## System Composition

The package provides building blocks for complex optical systems. Sources can be `Trainable` or `Static` scalar field sources. Modulators include phase masks, amplitude masks, and Thin Element Approximation diffractive elements. System composition uses pipe syntax (`source |> component1 |> component2 |> ...`) for intuitive construction. `FieldProbe` objects capture intermediate field states, enabling construction of loss functions that depend on fields at multiple planes, as well as visualization and debugging. Adjacent non-trainable components with compatible types can be merged for efficiency.

## Multi-Wavelength and Multimode Support

FluxOptics.jl natively supports polychromatic and multimode propagation.

## GPU Acceleration

Seamless GPU acceleration via CUDA.jl:
```julia
using CUDA
u = cu(ScalarField(...))  # Move to GPU
source = ScalarSource(u)   # Source inherits GPU context from u
doe = Phase(u, ...)        # Components inherit GPU context from u
system = source |> doe |> propagator  # Define an optical system
output_field, probes = system()  # GPU execution
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
