module Metrics

using ..Fields
using LinearAlgebra
export AbstractMetric, DotProduct, PowerCoupling
export SquaredFieldDifference, SquaredIntensityDifference
export compute_metric, backpropagate_metric

"""
    AbstractMetric

Abstract base type for optimization metrics in inverse optics design.

Metrics are callable objects that evaluate optimization objectives on optical fields.
They encapsulate target references and pre-allocated buffers for efficient gradient
computation during iterative optimization.

# Interface

Subtypes must implement:
- `compute_metric(m::AbstractMetric, u::NTuple{N, ScalarField})`: Evaluate metric
- `backpropagate_metric(m::AbstractMetric, u::NTuple{N, ScalarField}, ∂c)`: Compute gradients

# Callable Interface

All metrics are callable:
```julia
metric = PowerCoupling(target)
loss = metric(u)                    # Single field
losses = metric(u1, u2)             # Multiple fields
```

# Available Metrics

- [`DotProduct`](@ref): Complex overlap integral ⟨u,v⟩
- [`PowerCoupling`](@ref): Power coupled to target modes
- [`SquaredFieldDifference`](@ref): Field amplitude/phase matching
- [`SquaredIntensityDifference`](@ref): Intensity pattern matching

# Design Philosophy

- **Stateful**: Pre-allocate buffers at construction for zero-allocation evaluation
- **Flexible**: Support single or multiple fields, mode-selective or full matrices
- **Efficient**: Custom gradient implementations via `backpropagate_metric`
- **Composable**: Combine metrics for multi-objective optimization
"""
abstract type AbstractMetric end

"""
    compute_metric(m::AbstractMetric, u::NTuple{N, ScalarField})
    compute_metric(m::AbstractMetric, u::ScalarField)

Evaluate a metric on one or more optical fields.

This is the core evaluation function for metrics. In practice, prefer using the
callable interface `metric(u)` which calls this function internally.

# Arguments
- `m::AbstractMetric`: The metric to evaluate.
- `u`: Single `ScalarField` or tuple of `ScalarField`s.

# Returns
Array(s) containing metric values. Dimensions depend on the specific metric type
and whether `mode_selective=true`.

# Examples
```julia
metric = PowerCoupling(target)

# Callable interface (preferred)
loss = metric(u)

# Explicit call (equivalent)
loss = compute_metric(metric, u)

# Multiple fields
metric_multi = DotProduct(target1, target2)
losses = compute_metric(metric_multi, (u1, u2))
```

See also: [`backpropagate_metric`](@ref), [`AbstractMetric`](@ref)
"""
function compute_metric(m::AbstractMetric, u::NTuple{N, ScalarField}) where {N}
    error("Not implemented")
end

function compute_metric(m::AbstractMetric, u::ScalarField)
    first(compute_metric(m, (u,)))
end

"""
    (m::AbstractMetric)(u::Vararg{ScalarField})
    (m::AbstractMetric)(u::ScalarField)

Callable interface for metrics. Equivalent to `compute_metric(m, u)`.

This is the preferred way to evaluate metrics.

# Examples
```julia
metric = PowerCoupling(target)

# Single field
loss = metric(u)

# Multiple fields
metric_multi = DotProduct(target1, target2)
losses = metric_multi(u1, u2)
```

See also: [`compute_metric`](@ref)
"""
(m::AbstractMetric)(u::ScalarField) = compute_metric(m, u)
(m::AbstractMetric)(u::Vararg{ScalarField}) = compute_metric(m, u)

"""
    backpropagate_metric(m::AbstractMetric, u::NTuple{N, ScalarField}, ∂c)

Compute gradients of metric with respect to input fields.

Used internally by automatic differentiation. Most users don't need to call this directly.

# Arguments
- `m::AbstractMetric`: The metric.
- `u`: Tuple of input fields.
- `∂c`: Gradient w.r.t. metric output.

# Returns
Tuple of `ScalarField`s containing gradients w.r.t. each input field.
"""
function backpropagate_metric(m::AbstractMetric, u::NTuple{N, ScalarField}, ∂c) where {N}
    error("Not implemented")
end

function extra_dims(u::ScalarField{U, Nd}) where {U, Nd}
    ntuple(k -> k <= Nd ? 1 : size(u, k), ndims(u))
end

function split_size(u::ScalarField{U, Nd}) where {U, Nd}
    prod(size(u)[1:Nd]), prod(size(u)[(Nd + 1):end])
end

include("dot_product.jl")
include("power_coupling.jl")
include("squared_field_difference.jl")
include("squared_intensity_difference.jl")

end
