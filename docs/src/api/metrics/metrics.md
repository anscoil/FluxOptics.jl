# Metrics API

```@meta
CurrentModule = FluxOptics.Metrics
```

## Abstract Types

```@docs
AbstractMetric
```

## Core Functions

```@docs
(AbstractMetric)(::ScalarField)
compute_metric
backpropagate_metric
```

### Examples

```julia
using FluxOptics

# Create metric
target = ScalarField(target_data, (2.0, 2.0), 1.064)
metric = PowerCoupling(target)

# Evaluate metric
u = system()
loss = compute_metric(metric, u)

# Or use callable interface
loss = metric(u)

# With multiple fields
metric_multi = DotProduct(target1, target2)
losses = metric_multi(u1, u2)
```

## Dot Product Metrics

```@docs
DotProduct
```

### Examples

```julia
# Mode-selective dot product (default)
target = ScalarField(target_data, (64, 64, 3), (2.0, 2.0), 1.064)
metric = DotProduct(target; mode_selective=true)

u = system()
overlaps = metric(u)  # 1×1×3 array (per-mode)

# Full overlap matrix
metric_full = DotProduct(target; mode_selective=false)
overlap_matrix = metric_full(u)  # 3×3 matrix

# Multiple targets
metric_multi = DotProduct(target1, target2)
overlaps = metric_multi(u1, u2)
```

```@docs
PowerCoupling
```

### Examples

```julia
# Single target mode
target = ScalarField(target_data, (128, 128), (2.0, 2.0), 1.064)
normalize_power!(target)  # Not required - done internally

metric = PowerCoupling(target)
u = system()
power_coupled = metric(u)[]  # Scalar power in Watts

# Multi-mode coupling
modes_data = generate_mode_stack(layout, 128, 128, 2.0, 2.0, Gaussian(15.0))
modes = ScalarField(modes_data, (2.0, 2.0), 1.064)

metric = PowerCoupling(modes; mode_selective=true)
u = system()
powers = metric(u)  # Power per mode

# Use in optimization
function loss()
    u = system()
    -sum(metric(u))  # Maximize total coupled power
end
```

## Field Difference Metrics

```@docs
SquaredFieldDifference
```

### Examples

```julia
# Complex field matching
target = ScalarField(target_data, (128, 128), (2.0, 2.0), 1.064)

metric = SquaredFieldDifference(target)
u = system()
loss = metric(u)[]

# Multi-mode matching
target_multi = ScalarField(target_data, (64, 64, 3), (2.0, 2.0), 1.064)
metric_multi = SquaredFieldDifference(target_multi)
u_multi = system()
losses_per_mode = metric_multi(u_multi)  # 1×1×3 array
```

```@docs
SquaredIntensityDifference
```

### Examples

```julia
# Mode-by-mode intensity matching
target_intensity = abs2.(target_data)  # 128×128×3
u = ScalarField(ones(ComplexF64, 128, 128, 3), (2.0, 2.0), 1.064)

metric = SquaredIntensityDifference((u, target_intensity))
loss_per_mode = metric(u)  # 1×1×3 array

# Total intensity matching (summed over modes)
target_intensity_total = sum(abs2.(target_data), dims=3)[:,:,1]  # 128×128
metric_total = SquaredIntensityDifference((u, target_intensity_total))
loss_total = metric_total(u)[]  # Scalar

# Multiple field-target pairs
metric_multi = SquaredIntensityDifference(
    (field1, target1),
    (field2, target2)
)
losses = metric_multi(field1, field2)
```

## Optimization Patterns

### Basic Optimization

```julia
using Zygote, Optimisers

# Setup system and metric
system = source |> phase_mask |> propagator
target = ScalarField(target_data, (2.0, 2.0), 1.064)
metric = PowerCoupling(target)

# Loss function
function loss()
    u = system()
    -metric(u)[]  # Maximize coupling (minimize negative)
end

# Optimize
params = Functors.fmap(trainable, system)
opt_state = Optimisers.setup(Optimisers.Adam(0.01), params)

for iter in 1:100
    grads = gradient(loss, params)[1]
    Optimisers.update!(opt_state, params, grads)
end
```

### Multi-Objective Optimization

```julia
# Combine multiple metrics
target_mode = ScalarField(mode_data, (2.0, 2.0), 1.064)
target_intensity = desired_intensity_pattern

metric_coupling = PowerCoupling(target_mode)
metric_intensity = SquaredIntensityDifference((target_mode, target_intensity))

function combined_loss()
    u = system()
    α = 0.7  # Weighting factor
    α * (1.0 - metric_coupling(u)[]) + (1-α) * metric_intensity(u)[]
end
```

### Mode-Selective Optimization

```julia
# Optimize specific modes
modes = ScalarField(mode_stack, (2.0, 2.0), 1.064)
metric = PowerCoupling(modes; mode_selective=true)

function selective_loss()
    u = system()
    powers = metric(u)
    
    # Weight different modes
    weights = [1.0, 0.5, 0.2]  # Prioritize first mode
    -sum(reshape(weights, 1, 1, 3) .* powers)
end
```

## Technical Notes

### Metric Types
- All metrics are subtypes of `AbstractMetric`
- Metrics are callable: `metric(field)` calls `compute_metric`
- Support single or multiple field arguments

### Mode Selectivity
- `mode_selective=true`: Compute per-mode metrics (diagonal)
- `mode_selective=false`: Compute full coupling matrix
- Affects output dimensionality and computational cost

### Normalization
- `PowerCoupling` normalizes targets internally (copies first)
- `DotProduct` uses fields as-is
- Consider normalizing input fields before metrics

### Gradient Computation
- Metrics implement custom `backpropagate_metric` for efficiency
- Compatible with Zygote and other AD libraries
- Gradients flow through metric computations

### Performance
- Metrics cache internal buffers to avoid allocations
- Pre-compute metrics once, reuse for multiple evaluations
- Use in-place operations where possible

### Memory Layout
- Metrics store internal arrays matching field structure
- Buffers allocated at construction time
- GPU-compatible when using CuArrays

## Advanced Usage

### Custom Metric

```julia
# Define new metric type
struct MyMetric <: AbstractMetric
    reference::ScalarField
    buffer::Array{ComplexF64}
end

# Implement required methods
function compute_metric(m::MyMetric, u::Tuple{ScalarField})
    # Your metric computation
end

function backpropagate_metric(m::MyMetric, u::Tuple{ScalarField}, ∂c)
    # Your gradient computation
end
```

### Combining Metrics

```julia
# Wrapper for weighted sum
struct WeightedSum{M<:Tuple} <: AbstractMetric
    metrics::M
    weights::Vector{Float64}
end

function compute_metric(m::WeightedSum, u::NTuple{N, ScalarField}) where N
    sum(w * compute_metric(metric, u) for (w, metric) in zip(m.weights, m.metrics))
end
```

## See Also

- [OptimisersExt](../optimisers/index.md) for optimization algorithms
