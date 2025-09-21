module Metrics

using ..Fields
using LinearAlgebra
export AbstractMetric, DotProduct, PowerCoupling
export SquaredFieldDifference, SquaredIntensityDifference
export compute_metric, backpropagate_metric

abstract type AbstractMetric end

function compute_metric(m::AbstractMetric, u::NTuple{N, ScalarField}) where {N}
    error("Not implemented")
end

function compute_metric(m::AbstractMetric, u::ScalarField)
    first(compute_metric(m, (u,)))
end

(m::AbstractMetric)(u::Vararg{ScalarField}) = compute_metric(m, u)
(m::AbstractMetric)(u::ScalarField) = compute_metric(m, u)

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
