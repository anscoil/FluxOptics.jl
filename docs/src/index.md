# FluxOptics.jl

*Differentiable optical propagation and inverse design in Julia*

FluxOptics.jl is a Julia package for simulating scalar optical field propagation with full support for automatic differentiation. Design and optimize optical systems through gradient-based methods.

## Features

- 🌊 **Scalar field propagation**: Angular Spectrum, Rayleigh-Sommerfeld, Collins integral, Beam Propagation Method
- 🎭 **Optical components**: Phase masks, amplitude masks, DOEs, graded-index media
- 🎯 **Optimization ready**: Fully differentiable with Zygote/Enzyme support via ChainRulesCore
- 🔧 **Proximal operators**: TV regularization, sparsity, constraints
- 📊 **Built-in metrics**: Power coupling, field matching, intensity shaping
- 🚀 **GPU support**: CUDA acceleration available

## Quick Example

```julia
using FluxOptics

# Create a Gaussian beam
u = ScalarField(ones(ComplexF64, 128, 128), (2.0, 2.0), 1.064)
gaussian = Gaussian(20.0)
xv, yv = spatial_vectors(128, 128, 2.0, 2.0)
u.electric .= gaussian(xv, yv)

# Propagate through an optical system
source = ScalarSource(u)
phase_mask = Phase(u, (x, y) -> 0.01*(x^2 + y^2))
propagator = ASProp(u, 1000.0)  # 1 mm propagation

system = source |> phase_mask |> propagator
result = system()
```

## Documentation Structure

### [API Reference](api/index.md)
Complete documentation of all modules, types, and functions:
- **[GridUtils](api/gridutils/index.md)**: Coordinate systems and transformations
- **[Modes](api/modes/index.md)**: Gaussian beams, HG/LG modes, spatial layouts
- **[Fields](api/fields/index.md)**: ScalarField type and field operations
- **[Optical Components](api/optical_components/index.md)**: Propagators, masks, sources, systems
- **[OptimisersExt](api/optimisers/index.md)**: Optimization rules and proximal operators
- **[Metrics](api/metrics/index.md)**: Loss functions for inverse design

### Tutorials *(coming soon)*
Step-by-step guides for common use cases

## Getting Help

- 📖 Browse the [API Reference](api/index.md) for detailed function documentation
- 💬 Open an issue on [GitHub](https://github.com/anscoil/FluxOptics.jl) for bugs or feature requests
- 📧 Contact the maintainers for questions

## Citation

If you use FluxOptics.jl in your research, please cite:

```bibtex
@software{fluxoptics2025,
  author = {Barré, Nicolas},
  title = {FluxOptics.jl: Differentiable Optical Simulations},
  year = {2025},
  url = {https://github.com/anscoil/FluxOptics.jl}
}
```
