# API Reference

Complete documentation for all FluxOptics.jl modules and functions.

## Module Overview

FluxOptics is organized into focused modules for different aspects of optical simulation and inverse design.

### Foundation

**[GridUtils](gridutils/index.md)** - Coordinate systems and transformations  
Creates coordinate systems for evaluating optical modes and components on spatial grids. Provides 2D transformations (translations, rotations) and coordinate composition.

**[Modes](modes/index.md)** - Optical mode generation  
Generates common optical beam profiles (Gaussian, Hermite-Gaussian, Laguerre-Gaussian) and spatial layouts for multi-mode configurations. Includes speckle generation with controlled statistics.

**[Fields](fields/index.md)** - Field representation and operations  
Provides the `ScalarField` type, the central data structure representing optical fields with associated grid information, wavelength, and propagation direction. Supports multi-wavelength, tilt tracking, and field operations.

### Optical Components

**[Optical Components](optical_components/index.md)** - Building blocks for optical systems

The heart of FluxOptics, providing all optical elements and system composition tools.

**Core infrastructure:**
- **[Core](optical_components/core/index.md)** - Abstract component hierarchy, trainability system (Static/Trainable/Buffered), and bidirectional propagation interface

**Creating and modifying fields:**
- **[Sources](optical_components/sources/index.md)** - Generate initial optical fields (e.g., `ScalarSource`). Sources are trainable, allowing optimization of input beam profiles
- **[Modulators](optical_components/modulators/index.md)** - Modify field amplitude and phase: `Phase` for pure phase modulation, `Mask` for amplitude/complex transmission, `TeaDOE` for diffractive elements
- **[Fourier](optical_components/fourier/index.md)** - Frequency-domain operations. `FourierWrapper` applies components in Fourier space, while `FourierPhase` and `FourierMask` provide convenient filtering

**Propagating fields:**
- **[Free-Space Propagators](optical_components/freespace/index.md)** - Field propagation through homogeneous media: Angular Spectrum (ASProp), Rayleigh-Sommerfeld (RSProp), Collins integral for ABCD systems, Fourier lenses
- **[Bulk Propagators](optical_components/bulk/index.md)** - Beam Propagation Method (BPM) for inhomogeneous media with spatially-varying refractive index. Supports paraxial and non-paraxial tilted propagation
- **[Active Media](optical_components/active/index.md)** - Stationary gain and amplification with saturable gain sheets

**Building systems:**
- **[System](optical_components/system/index.md)** - System construction using pipe operator `|>` to chain components. Fully differentiable and callable. `FieldProbe` captures intermediate fields for custom objectives, visualization, and debugging

**Advanced utilities:**
- **[Utilities](optical_components/utilities/index.md)** - Helper components: `PadCropOperator` for aliasing-free Fourier-based convolution, `TiltAnchor` for off-axis beam tracking, `BasisProjectionWrapper` for reduced-parameter optimization

### Optimization

**[OptimisersExt](optimisers/index.md)** - Optimization algorithms and proximal operators  
Custom optimization rules (Descent, Momentum, FISTA) and proximal operators for constrained optimization. Use `make_rules` for per-component learning rates, `ProxRule` for regularization (TV, sparsity, constraints). Integration with Optimisers.jl ecosystem.

**[Metrics](metrics/index.md)** - Loss functions for inverse design  
Field overlap metrics (DotProduct, PowerCoupling) for mode matching, field and intensity matching objectives (SquaredFieldDifference, SquaredIntensityDifference). Custom gradient implementations for efficiency.

### Visualization

**[Plotting](plotting/index.md)** - Visualization tools  
Field visualization with multiple representations (intensity, phase, real/imaginary, complex). Stack visualization with animated sliders. Component visualization for phase masks and DOEs. Requires Makie.jl backend (CairoMakie, GLMakie, or WGLMakie).

## Design Philosophy

FluxOptics follows these principles:

- **Differentiable by design**: All components work with automatic differentiation
- **GPU-ready**: Seamless CUDA.jl integration for acceleration
- **Composable**: Build complex systems from simple components using pipe operator
- **Efficient**: Pre-allocated buffers on-demand and optimized kernels
- **Flexible**: Support for multi-wavelength, multi-mode, and off-axis propagation

## Typical Workflow: Beam Splitter

This example demonstrates inverse design of a cascaded diffractive optical system that splits a single Gaussian beam into four spots arranged in a square pattern. We use a similar setup to the [TorchOptics beam splitter example](https://torchoptics.readthedocs.io/en/stable/auto_examples/training_diffractive_splitter.html) to enable comparison of both performance and API design philosophy.

### 1. Setup: Input and Target modes

```@example splitter
using FluxOptics, Zygote, Optimisers
using CairoMakie

# Input: single Gaussian mode
ds = 10.0  # 10 microns
xv, yv = spatial_vectors(250, 250, ds, ds)
w0 = 150.0  # 150 microns
λ = 0.7  # 700 nm
u0 = ScalarField(Gaussian(w0)(xv, yv), (ds, ds), λ)
normalize_power!(u0)  # normalize u0 to unit power

# Target: coherent superposition of four Gaussian modes
offset = 3.8 * w0
positions = [(-offset, -offset), (offset, -offset), (-offset, offset), (offset, offset)]
vf = sum(positions) do (Δx, Δy)
    ScalarField(Gaussian(w0)(xv, yv, Shift2D(Δx, Δy)), (ds, ds), λ)
end
normalize_power!(vf)  # normalize vf to unit power

# Visualize input and target
visualize(((u0, vf),), intensity; colormap=:inferno, height=120)
```

### 2. Build Trainable System

We use a cascade of three trainable phase masks with Rayleigh-Sommerfeld propagation between them:

```@example splitter
source = ScalarSource(u0)
doe1 = Phase(u0, (x, y) -> 0.0; trainable=true, buffered=true)
doe2 = Phase(u0, (x, y) -> 0.0; trainable=true, buffered=true)
doe3 = Phase(u0, (x, y) -> 0.0; trainable=true, buffered=true)
prop = RSProp(u0, 2e5; use_cache=true)  # 200mm propagation

system = source |> doe1 |> prop |> doe2 |> prop |> doe3 |> prop |> (; inplace=true)
nothing # hide
```

### 3. Define Loss and Optimize

We maximize power coupling to the target pattern using the Adam optimizer:

```@example splitter
metric = PowerCoupling(vf)
loss(m) = sum(1 .- metric(m().out))

opt = FluxOptics.setup(Optimisers.Adam(0.1), system)

# Warm up for accurate allocation estimation
_, g = withgradient(loss, system);
FluxOptics.update!(opt, system, g[1])

# Reset phase masks (allows re-running optimization)
foreach(d -> fill!(d, 0), (doe1, doe2, doe3))

# Setup optimizer
losses = Float64[]

# Optimization loop
mem = @allocated for iter in 1:200
    val, grads = withgradient(loss, system)
    FluxOptics.update!(opt, system, grads[1])
    push!(losses, val)
end

println("Memory: ", Base.format_bytes(mem))
```

### 4. Results

#### Convergence

```@example splitter
fig = Figure(size=(300, 250))
ax = Axis(fig[1, 1], 
    yscale=log10, 
    xlabel="Iteration", 
    ylabel="Loss",
    title="Optimization Convergence"
)
lines!(ax, losses; linewidth=2)
fig
```

#### Final Output Field

```@example splitter
uf = system().out
visualize(uf, (intensity, x -> -phase(x)); 
    colormap=(:inferno, twilight_shifted), height=120)
```

#### Optimized Phase Masks

```@example splitter
# Extract and visualize the three DOE phase profiles
doe_phases = hcat(map(d -> -angle.(cis.(collect(d))), (doe1, doe2, doe3)))

visualize(doe_phases, identity;  colormap=twilight_shifted, height=150)
```

### Performance Benchmark

**Hardware:** NVIDIA RTX 4070 Super (12GB VRAM)

**CPU Performance (multi-threaded):**
- Time: ~5 seconds (200 iterations)
- Memory: **41 MiB**

**GPU Performance:**
- Time: ~0.27 seconds (200 iterations)  
- GPU Memory: Minimal allocation

**Comparison with [TorchOptics](https://torchoptics.readthedocs.io/en/stable/auto_examples/training_diffractive_splitter.html):**
- CPU: TorchOptics ~7s vs FluxOptics ~5s (1.4× faster)
- GPU: TorchOptics ~3.5s vs FluxOptics ~0.27s (13× faster)

**Scalability:** The low memory footprint (41 MiB) enables optimization of much larger problems. FluxOptics can handle larger grids, multi-wavelength, and multi-mode problems that exceed GPU memory limits in other frameworks. This qualitative difference—not just speed, but **problem accessibility**—is the key advantage for real-world optical design.

### GPU Acceleration

To run on GPU, add these lines at the beginning:

```julia
using CUDA
CUDA.allowscalar(false)

# After creating fields
u0 = cu(u0)
vf = cu(vf)
```

All operations will automatically run on GPU with no other code changes required.

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

**Utilities** contains helper components: `PadCropOperator` for aliasing-free Fourier-based convolution (pad → convolve → crop), `TiltAnchor` for off-axis beam tracking, `BasisProjectionWrapper` for reduced-parameter optimization.

**Free-Space Propagators** implement field propagation through homogeneous media: Angular Spectrum method (ASProp), Rayleigh-Sommerfeld diffraction (RSProp), Collins integral for ABCD systems, Fourier lenses.

**Bulk Propagators** use Beam Propagation Method (BPM) for inhomogeneous media with spatially-varying refractive index. Supports paraxial  and non-paraxial tilted propagation.

**Active Media** models stationary gain and amplification with saturable gain sheets.

### Optimization Modules

**OptimisersExt** provides optimization algorithms and proximal operators. Use `make_rules` for per-component learning rates, `ProxRule` for constrained optimization with regularization.

**Metrics** defines loss functions for inverse design: `PowerCoupling` for mode matching, `SquaredFieldDifference` for field shaping, `SquaredIntensityDifference` for intensity targets.
