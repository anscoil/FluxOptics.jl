# FluxOptics.jl Test Suite

This directory contains the test suite for FluxOptics.jl, organized following the documentation structure.

## Running Tests

### Run all tests
```julia
julia> using Pkg
julia> Pkg.test("FluxOptics")
```

Or from the command line:
```bash
julia --project -e 'using Pkg; Pkg.test()'
```

### Run specific test file
```julia
julia> include("test/gridutils_test.jl")
```

## Test Structure

Tests are organized by module, following the documentation structure:

- **`gridutils_test.jl`**: Spatial/frequency vectors, meshgrids, rotations, coordinate transformations
- **`modes_test.jl`**: Gaussian, Hermite-Gaussian, Laguerre-Gaussian mode generation and properties
- **`fields_test.jl`**: ScalarField construction, operations, normalization, power calculations
- **`optical_components_test.jl`**: 
  - Sources (ScalarSource)
  - Modulators (Phase, Mask, TeaDOE)
  - Propagators (ASProp, RSProp, ParaxialProp, CollinsProp, BPM)
  - Utilities (pad, crop, probes)
  - System composition
  - Active media (GainSheet)
- **`optimisers_test.jl`**: Proximal operators (ISTA, Clamp, Positive), optimization rules (Fista, ProxRule)
- **`metrics_test.jl`**: Coupling efficiency, power coupling, loss functions

## Key Test Categories

### Unitarity Tests
Propagators are tested for unitarity: Forward → Backward propagation should return the original field.

```julia
# Example: AS propagator unitarity
u_prop = propagate(u, prop, Forward)
u_back = propagate(u_prop, prop, Backward)
@test coupling_efficiency(u, u_back) > 0.9999
```

### Energy Conservation
Power should be conserved through lossless propagation.

### Normalization
Mode generation and field operations preserve correct normalization.

### Differentiation
Loss functions and metrics are tested for differentiability with Zygote.

## Adding New Tests

When adding new functionality:

1. Add tests to the appropriate file based on module
2. Follow the existing `@testset` structure
3. Test both functionality and properties (unitarity, energy conservation, etc.)
4. Include gradient checks for differentiable operations

