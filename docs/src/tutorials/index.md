# Tutorials

These tutorials demonstrate various applications of FluxOptics.jl, from basic wave propagation to advanced inverse design problems.

## Documentation vs Notebooks

The tutorials are available in two formats:

- **Documentation pages** (this site): Read-only versions with embedded figures and explanations
- **Interactive Jupyter notebooks** ([`notebooks/` folder](https://github.com/anscoil/FluxOptics.jl/tree/main/notebooks)): Executable versions including execution times and interactive exploration

!!! note "Execution Times"
    The interactive notebooks include timing information for each tutorial. All tutorials are GPU-optimized and execute very quickly on modern hardware (GeForce RTX 4070 Super):
    
    - **Tutorials 1-4**: 2-6 seconds
    - **Tutorial 5**: ~42 seconds for the full 105-mode example
	- **Tutorial 6**: ~34 seconds for the conversion of 45 modes with 8 planes
    
    CPU execution is possible but significantly slower and not recommended for iterative development.

## Running the Tutorials

To run the notebooks yourself:

1. Install FluxOptics.jl and required dependencies
2. Clone the repository:
   ```julia
   git clone https://github.com/anscoil/FluxOptics.jl.git
   cd FluxOptics.jl/notebooks
   ```
3. Launch Jupyter and open the desired notebook

!!! warning "GPU Recommended"
    These tutorials are optimized for GPU execution. Ensure CUDA.jl is properly configured for the timing benchmarks shown above. CPU execution will be significantly slower.
    
    **For CPU-only users**: The notebooks include comments indicating which lines to modify for CPU execution (typically CUDA array conversions).

## Available Tutorials

### [FoxLi Simulation](01_FoxLi_simulation.md)
Demonstrates basic wave propagation and saturated gain using the Fox-Li algorithm to simulate resonator modes.

### [Field Retrieval](02_field_retrieval.md)
Shows how to retrieve complex field information from intensity measurements using phase retrieval techniques.

### [Multi-Wavelength Beam Shaping](03_RGB_beam_shaping.md)
Designs optical systems that shape beams at multiple wavelengths simultaneously (RGB).

### [Waveguide Tomography](04_waveguide_tomography.md)
Demonstrates inverse reconstruction of waveguide refractive index profiles from tomographic intensity measurements using plane wave illumination at multiple angles.

### [Multimode Intensity Shaping](05_multimode_intensity_shaping.md)
Optimizes multimode fiber systems to achieve target intensity distributions. This tutorial demonstrates optimization with 105 modes and is the most computationally intensive example.

### [Hermite-Gaussian Multimode Sorter](06_multimode_HG_sorter.md)
Designs a 45-mode sorter transforming triangular Gaussian arrays into copropagating Hermite-Gaussian modes with 8 cascaded phase masks. Revisits the seminal Carpenter & Fontaine approach (CLEO/Europe 2017) using algorithmic differentiation instead of classical error-reduction algorithms.
