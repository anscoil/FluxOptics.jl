# FluxOptics.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://anscoil.github.io/FluxOptics.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://anscoil.github.io/FluxOptics.jl/dev/)
[![Build Status](https://github.com/anscoil/FluxOptics.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/anscoil/FluxOptics.jl/actions/workflows/CI.yml?query=branch%3Amain)

**Differentiable optical propagation and inverse design in Julia**

FluxOptics.jl is a high-performance Julia package for simulating scalar optical field propagation with full support for automatic differentiation. Design and optimize complex optical systems using gradient-based methods.

## ✨ Features

- 🌊 **Multiple propagation methods**: Angular Spectrum, Rayleigh-Sommerfeld, Collins integral, Beam Propagation Method
- 🎭 **Rich component library**: Phase masks, amplitude masks, diffractive optical elements (DOEs)
- 🎯 **Optimization-first design**: Fully differentiable with Zygote and Enzyme via ChainRulesCore
- 🔧 **Advanced regularization**: Proximal operators for constrained optimization
- 📊 **Built-in metrics**: Power coupling, field overlap, intensity matching
- 🚀 **GPU acceleration**: Seamless CUDA.jl integration for large-scale simulations
- 🧩 **Composable systems**: Build complex optical systems with intuitive piping syntax

## 📦 Installation

```julia
using Pkg
Pkg.add("FluxOptics")
```

Or from the Julia REPL, press `]` to enter package mode:
```
add FluxOptics
```

## 🚀 Quick Start

See the [documentation](https://anscoil.github.io/FluxOptics.jl/stable/) for detailed examples and tutorials.

## 📚 Documentation

Full documentation is available at [anscoil.github.io/FluxOptics.jl](https://anscoil.github.io/FluxOptics.jl/stable/)

### API Reference

- **[GridUtils](https://anscoil.github.io/FluxOptics.jl/stable/api/gridutils/)**: Coordinate systems and transformations
- **[Modes](https://anscoil.github.io/FluxOptics.jl/stable/api/modes/)**: Gaussian, Hermite-Gaussian, Laguerre-Gaussian modes
- **[Fields](https://anscoil.github.io/FluxOptics.jl/stable/api/fields/)**: ScalarField type and operations
- **[OpticalComponents](https://anscoil.github.io/FluxOptics.jl/stable/api/optical_components/)**: Propagators, masks, sources, systems
- **[Optimisers](https://anscoil.github.io/FluxOptics.jl/stable/api/optimisers/)**: Optimization rules and proximal operators
- **[Metrics](https://anscoil.github.io/FluxOptics.jl/stable/api/metrics/)**: Loss functions for inverse design

## 🎯 Use Cases

FluxOptics.jl is designed for:

- Diffractive and spatially-varying optical element design
- Fiber coupling efficiency maximization  
- Beam shaping and intensity distribution control
- Wavefront analysis and aberration correction
- Phase and field retrieval from intensity measurements
- Multimode beam characterization and decomposition
- Computer-generated holography
- Stationary transverse mode analysis and design of laser resonators
- Optical tomography and waveguide characterization

## 🛠️ Key Capabilities

### Multi-Wavelength Support

```julia
# Propagate multiple wavelengths simultaneously
λs = [0.8, 1.064, 1.55]
u_multi = ScalarField(data, (2.0, 2.0), λs)
propagator = ASProp(u_multi, 1000.0)
result = propagate(u_multi, propagator, Forward)
```

### Off-Axis Propagation

```julia
# Track tilted beams through optical systems
u_tilted = ScalarField(data, (2.0, 2.0), 1.064; tilts=(0.05, 0.02))
propagator = ASProp(u_tilted, 1000.0; track_tilts=true)
result = propagate(u_tilted, propagator, Forward)
```

### GPU Acceleration

```julia
using CUDA

# Move field to GPU
u_gpu = cu(u)

# All operations work seamlessly on GPU
propagator = ASProp(u_gpu, 1000.0)
result_gpu = propagate(u_gpu, propagator, Forward)
```

### Constrained Optimization

```julia
# Apply constraints during optimization with proximal operators
prox_rule = ProxRule(Optimisers.Adam(0.01), constraint_function)
opt_state = setup(prox_rule, params)
update!(opt_state, params, grads)
```

## 🤝 Feedback and Suggestions

Have ideas for new features or use cases? Open an issue on [GitHub](https://github.com/anscoil/FluxOptics.jl/issues) to discuss!

I'm particularly interested in:
- Real-world application needs and use cases
- Performance bottlenecks in your workflows
- Missing features for your research

For bug reports, please include a minimal reproducible example.

## 📝 Citation

If you use FluxOptics.jl in your research, please cite:

```bibtex
@software{fluxoptics2025,
  author = {Barré, Nicolas},
  title = {FluxOptics.jl: Differentiable Optical Simulations in Julia},
  year = {2025},
  url = {https://github.com/anscoil/FluxOptics.jl},
  version = {0.1.0}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

FluxOptics.jl builds upon the excellent Julia ecosystem, particularly:
- [Zygote.jl](https://github.com/FluxML/Zygote.jl) for automatic differentiation
- [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl) for defining custom backpropagation rules
- [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) for optimization algorithms
- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) for GPU acceleration
- [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) for documentation

---

**Maintainer**: Nicolas Barré ([@anscoil](https://github.com/anscoil))

**Status**: Active development
