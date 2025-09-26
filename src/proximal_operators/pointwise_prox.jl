"""
    PointwiseProx(f)

Apply a function pointwise as a proximal operator.

Creates a proximal operator that applies function `f` element-wise to the parameter
array. The function should take the current parameter value and return the
projected/regularized value.

# Arguments
- `f`: Function to apply element-wise

# Examples
```jldoctest
julia> clamp_prox = PointwiseProx(x -> clamp(x, -1, 1));

julia> threshold_prox = PointwiseProx(x -> abs(x) < 0.1 ? 0 : x);

julia> model = randn(10, 10);

julia> prox_state = ProximalOperators.init(clamp_prox, model);

julia> ProximalOperators.apply!(clamp_prox, prox_state, model);

julia> all(x -> -1 ≤ x ≤ 1, model)  # All values clamped
true
```

See also: [`ClampProx`](@ref), [`PositiveProx`](@ref), [`ProxRule`](@ref)
"""
struct PointwiseProx{F} <: AbstractProximalOperator
    f::F
    function PointwiseProx(f::F) where {F}
        new{F}(f)
    end
end

init(prox::PointwiseProx, x::AbstractArray) = ()

function apply!(prox::PointwiseProx, state, x::AbstractArray)
    @. x = prox.f(x, state...)
end

include("ista_prox.jl")
include("clamp_prox.jl")
include("positive_prox.jl")
