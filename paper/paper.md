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
date: 18 February 2026
bibliography: paper.bib
---

# Summary

FluxOptics.jl is a Julia package for simulating optical field propagation with full support for automatic differentiation. It enables gradient-based inverse design of optical components, which involves determining the structure of an optical element (e.g., lens, diffraction grating, phase mask) that produces a desired light pattern or functionality. The package implements scalar wave propagation methods that are computationally efficient alternatives to finite-difference time-domain (FDTD) simulations, particularly suited for applications in holography, additive manufacturing, and optical characterization.

FluxOptics.jl provides multiple propagation algorithms for free-space and graded-index media, a composable architecture for building complex optical systems, GPU acceleration, and optimization tools including proximal operators for constrained inverse design. The architecture supports current scalar field applications and is designed to extend to vector field propagation for polarization-dependent components and dielectric metasurfaces.

# Statement of Need

Inverse design of optical components has become increasingly important with the rise of freeform optics [@Schmidt2020; @Barre2025], diffractive optical elements (DOEs) [@Dinc2020], and metasurfaces [@Molesky2018; @Peurifoy2018]. Traditional optimization approaches include gradient-free methods (evolutionary algorithms, Bayesian optimization, stochastic search), which can be effective for low-dimensional problems with up to hundreds of parameters. However, these methods become intractable for spatially-resolved optical elements with thousands or millions of parameters. Gradient-based optimization using automatic differentiation [@Hughes2018; @Minkov2020] enables efficient convergence at a computational cost comparable to a single forward simulation.

However, existing tools face several limitations. Full-wave electromagnetic solvers like FDTD provide high accuracy but are computationally prohibitive for optimization, often requiring hours per forward simulation and limited to 2D or small 3D domains [@Oskooi2010]. Python packages like TorchOptics [@TorchOptics] provide differentiable scalar wave propagation but suffer from performance bottlenecks. In Julia, WaveOpticsPropagation.jl [@Wechsler:24] offers individual propagation functions, while FluxOptics.jl provides an integrated inverse design framework combining unified propagation components with established optimization methodology (proximal operators, FISTA acceleration) for end-to-end optical system design.

FluxOptics.jl addresses these gaps through several key innovations. Implemented in Julia [@Bezanson2017] with custom adjoint rules via ChainRulesCore.jl, the package achieves high-performance CPU and GPU implementations that outperform TorchOptics, particularly on GPU (13× speedup), while enabling seamless integration with Zygote.jl [@Innes2019] for automatic differentiation.

The package provides an extensible component architecture designed for users to implement their own optical components. The interface offers two implementation patterns depending on performance requirements: rapid prototyping with automatic adjoint derivation, or fine-grained control over memory allocations and gradient computation for production-level performance.

FluxOptics.jl emphasizes composability. Optical systems are built using Julia's pipe operator (`|>`), allowing intuitive construction of cascaded systems. The `FieldProbe` mechanism enables capturing intermediate field states for multi-objective optimization, visualization, and debugging.

The package implements efficient propagation methods through kernel caching strategies that avoid redundant computations. This makes scalar wave propagation practical for iterative optimization in applications such as additive manufacturing and intensity-based waveguide tomography.

Finally, the architecture is designed for extensibility from scalar to vector field propagation, with planned support for polarization-dependent components.

FluxOptics.jl emerged from practical research challenges in laser cavity design [@Barre2014], waveguide characterization [@Barre2021], and multimode beam control [@Barre2022OL; @Barre2022CIRP]. The package provides a unified framework with consistent API design that consolidates these diverse applications and enables reproducing established results such as multimode light conversion [@Fontaine2019] using modern gradient-based optimization.

# Key Features

## Component Architecture and Automatic Differentiation

All optical components inherit from a unified abstract type hierarchy that enables automatic differentiation. Components are divided into sources (generating optical fields) and pipe components (transforming fields). Connecting components with the pipe operator (`|>`) creates an `OpticalSystem` that executes the complete optical simulation when invoked.

The architecture offers two implementation patterns. **Pure components** require only a pure `propagate` method where automatic differentiation automatically derives adjoints, enabling rapid prototyping. **Custom components** implement the full interface with manual adjoint specification (leveraging ChainRulesCore.jl internally), providing fine-grained control over memory allocation and computational efficiency.

The package provides optimization tools built on Optimisers.jl including proximal operators and FISTA acceleration [@Beck2009].

## Optical Components and Field Propagation

FluxOptics.jl provides optical components including sources, modulators, and propagation methods. **Free-space propagation** includes Angular Spectrum Method, Rayleigh-Sommerfeld diffraction, and Collins integral for ABCD systems, with native support for tilted beam propagation. **Graded-index media** are handled through Beam Propagation Method with both paraxial and non-paraxial formulations. **Fourier optics** capabilities include Fourier lenses and frequency-domain filtering. **Active media** can be simulated using stationary gain sheets with saturable amplification.

## System Composition

System composition uses pipe syntax (`source |> component1 |> component2 |> ...`) for intuitive construction. Sources can be `Trainable` or `Static`. Modulators include phase masks, amplitude masks, and Thin Element Approximation (TEA) diffractive elements. `FieldProbe` objects capture intermediate field states for multi-objective optimization and debugging.

## Multi-Wavelength and Multimode Support

FluxOptics.jl natively supports polychromatic and multimode propagation.

## GPU Acceleration

Seamless GPU acceleration via CUDA.jl with automatic context propagation through components.

## Performance Benchmark

We compared FluxOptics.jl with TorchOptics on a [beam splitter inverse design task](https://anscoil.github.io/FluxOptics.jl/stable/api/#Typical-Workflow:-Beam-Splitter) (250×250 grid, 3 DOEs, 200 optimization iterations):

| Platform | TorchOptics | FluxOptics.jl | Speedup |
|----------|-------------|---------------|---------|
| CPU (multi-threaded) | ~7s | ~5s | 1.4× |
| GPU (NVIDIA RTX 4070 Super) | ~3.5s | ~0.27s | 13× |

CPU memory footprint: 41 MiB total allocation for 200 iterations (~205 KiB per iteration), enabling scaling to large multimode problems.

# Tutorials and Documentation

FluxOptics.jl provides comprehensive documentation including six tutorials:

1. **Fox-Li Cavity Simulation**: Laser cavity eigenmodes in semi-degenerate resonators
2. **Field Retrieval from Intensity**: Reconstructing complex optical fields from intensity-only measurements, generalizing classical iterative projection methods [@Fienup1982]
3. **Multi-Wavelength Beam Shaping**: Chromatic DOE cascades for independent control of red, green, and blue beams
4. **Waveguide Tomography**: Reconstructing refractive index profiles from angle-resolved intensity data
5. **Multimode Intensity Shaping**: Shaping 105 Laguerre-Gaussian modes using 2 cascaded DOEs with TV-norm regularization
6. **Hermite-Gaussian Multimode Sorter**: Converting 45 spatially separated Gaussian modes to copropagating higher-order HG modes

Complete API documentation is available at [https://anscoil.github.io/FluxOptics.jl/stable/](https://anscoil.github.io/FluxOptics.jl/stable/).

# References
