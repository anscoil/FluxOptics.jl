<p align="center">
  <img src="logo_large.svg" alt="FluxOptics.jl" width="600"/>
</p>

# FluxOptics.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://anscoil.github.io/FluxOptics.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://anscoil.github.io/FluxOptics.jl/dev/)
[![Build Status](https://github.com/anscoil/FluxOptics.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/anscoil/FluxOptics.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.09734/status.svg)](https://doi.org/10.21105/joss.09734)

**Differentiable optical propagation and inverse design in Julia**

FluxOptics.jl enables gradient-based optimization of optical systems through fully differentiable wave propagation. It allows you to design diffractive optical elements, optimize beam shaping, reconstruct optical fields, and characterize photonic structures.

## ✨ Key Features

- 🌊 **Wave propagation**: Angular Spectrum, Rayleigh-Sommerfeld, Beam Propagation Method
- 🎯 **Inverse design**: End-to-end optimization of phase masks, DOEs, and refractive index profiles
- 🔧 **Proximal optimization**: FISTA, TV regularization, ISTA sparsity, custom constraints
- 📊 **Multi-wavelength & multimode**: Polychromatic propagation and mode coupling
- 🚀 **GPU accelerated**: Seamless CUDA support
- 🧩 **Composable architecture**: Intuitive piping syntax for complex optical systems

## Applications

FluxOptics.jl is designed for:
- **Inverse design**: Diffractive optical elements, metasurfaces, beam shaping
- **Optical characterization**: Phase retrieval, tomographic reconstruction, waveguide/fiber analysis
- **Laser physics**: Cavity eigenmode analysis, mode selection, resonator design
- **Photonics**: Fiber coupling, multimode decomposition, GRIN media simulation

## 📦 Installation
```julia
using Pkg
Pkg.add("FluxOptics")
```

**Minimum Julia version**: 1.11

### Required for optimization workflows

To run gradient-based optimization, you need:

**Automatic differentiation**:
```julia
Pkg.add("Zygote")    # Used in all examples and tutorials
```

**Optimization algorithms** :
FluxOptics.jl reexports `Descent` from Optimisers.jl and provides `Fista` (Nesterov-accelerated proximal gradient). Install Optimisers.jl for additional algorithms:

```julia
Pkg.add("Optimisers")  # For additional optimizers (Adam, Momentum, etc.)
```

### Optional: Visualization

For plotting capabilities:
```julia
Pkg.add("CairoMakie")  # for visualize()
Pkg.add("GLMakie")     # for visualize_slider()
```

## 🚀 Quick Start

Design diffractive optical elements that split a Gaussian beam into two equal outputs:

```julia
using FluxOptics, Zygote

# Uncomment to use more FFTW threads
# using FFTW
# FFTW.set_num_threads(6)

# using CUDA  # Uncomment to use CUDA

# Setup: 512×512 grid, 1 μm pitch, 1.064 μm wavelength
ns = (512, 512)
ds = (1.0, 1.0)
λ = 1.064
x, y = spatial_vectors(ns, ds)

# Source and target fields
u0 = ScalarField(Gaussian(30.0)(x, y), ds, λ)
target = ScalarField(Gaussian(30.0)(x, y, Shift2D(-60, 0)), ds, λ) +
         ScalarField(Gaussian(30.0)(x, y, Shift2D(60, 0)), ds, λ)
# Uncomment if you are using CUDA
# u0 = cu(u0)
# target = cu(target)
normalize_power!(u0)
normalize_power!(target)

# Optical system: source → propagation → DOE → propagation → DOE → propagation
doe1 = Phase(u0; trainable=true, buffered=true)
doe2 = Phase(u0; trainable=true, buffered=true)
prop1 = RSProp(u0, 1500.0)
prop2 = RSProp(u0, 2000.0)
system = ScalarSource(u0) |> prop1 |> doe1 |> prop2 |> doe2 |> prop1

# Optimize
loss(system) = sum(abs2, system().out.electric - target.electric)
opt = FluxOptics.setup(Fista(50), system)

# Lower the number of iterations for a quick test on cpu
for i in 1:200
    l, ∇ = Zygote.withgradient(loss, system)
    FluxOptics.update!(opt, system, ∇[1])
end

# Result: 99.92% coupling efficiency
output = system().out
coupling_efficiency(output, target)
```

## 📖 Tutorials

**[Complete documentation →](https://anscoil.github.io/FluxOptics.jl/stable/)**

Six comprehensive tutorials:

| Tutorial | Description |
|----------|-------------|
| [**Fox-Li Cavity**](https://anscoil.github.io/FluxOptics.jl/stable/tutorials/01_FoxLi_simulation/) | Eigenmode analysis in semi-degenerate laser resonators |
| [**Field Retrieval**](https://anscoil.github.io/FluxOptics.jl/stable/tutorials/02_field_retrieval/) | Reconstruct amplitude and phase from intensity-only data |
| [**Multi-Wavelength**](https://anscoil.github.io/FluxOptics.jl/stable/tutorials/03_RGB_beam_shaping/) | Independent RGB beam control with cascaded DOEs |
| [**Waveguide Tomography**](https://anscoil.github.io/FluxOptics.jl/stable/tutorials/04_waveguide_tomography/) | Refractive index reconstruction from angle-resolved intensity |
| [**Multimode Shaping**](https://anscoil.github.io/FluxOptics.jl/stable/tutorials/05_multimode_intensity_shaping/) | Shape 105 modes into square/ring targets with TV regularization |
| [**Mode Sorting**](https://anscoil.github.io/FluxOptics.jl/stable/tutorials/06_multimode_HG_sorter/) | Transform 45 Gaussian beams into Hermite-Gaussian modes (8-plane system) |

## ⚡ Performance

FluxOptics.jl is benchmarked against
[JaxOptics](https://github.com/anscoil/jaxoptics),
[WaveOpticsPropagation.jl](https://github.com/JuliaPhysics/WaveOpticsPropagation.jl),
and [TorchOptics](https://github.com/MatthewFilipovich/torchoptics)
on Angular Spectrum propagation (512×512, up to 1386 modes, NVIDIA RTX 4070):

| Library | GPU speedup vs FluxOptics |
|---------|--------------------------|
| **FluxOptics** | **1×** (baseline) |
| **JaxOptics** | 0.8× |
| **WaveOpticsPropagation.jl** | 0.6× |
| **TorchOptics** | 0.002× |

FluxOptics also handles **~1.5× more modes** than JaxOptics before running out of GPU memory.

> **Note**: these benchmarks measure Angular Spectrum propagation performance in isolation.
> On full optimization loops, the gap may narrow: TorchOptics showed as little as 13×
> slowdown on a 3-plane monomode conversion task. For FFT and tensor-dominated problems,
> XLA-based frameworks like JAX may even slightly outperform Julia due to whole-graph
> compilation, and for such cases, JaxOptics may well be sufficient.
> FluxOptics is expected to show stronger advantages on more complex optical components
> where manual optimization and Julia's flexibility become decisive.

→ [Full benchmark details](benchmarks/README.md)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Reporting bugs and requesting features
- Sharing use cases and examples
- Submitting code contributions

## 📝 Citation

If you use FluxOptics.jl in your research, please cite:

```bibtex
@article{Barré2026:FluxOptics,
  doi = {10.21105/joss.09734},
  url = {https://doi.org/10.21105/joss.09734},
  year = {2026},
  publisher = {The Open Journal},
  volume = {11},
  number = {118},
  pages = {9734},
  author = {Barré, Nicolas},
  title = {FluxOptics.jl: A Differentiable Wave Optics Framework for Inverse Design in Julia},
  journal = {Journal of Open Source Software}
} 
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built on the Julia ecosystem:
- [Zygote.jl](https://github.com/FluxML/Zygote.jl) / [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl) for automatic differentiation
- [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) for optimization algorithms
- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) for GPU acceleration
