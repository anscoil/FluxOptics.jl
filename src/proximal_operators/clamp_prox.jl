"""
    ClampProx(lo, hi)

Clamp values to the range [lo, hi].

Simple box constraint proximal operator that projects parameters onto a box.
Useful for physically meaningful constraints (e.g., transmittance ∈ [0,1]).

# Arguments
- `lo`: Lower bound
- `hi`: Upper bound

# Examples
```jldoctest
julia> unit_clamp = ClampProx(0.0, 1.0);  # Unit interval

julia> phase_clamp = ClampProx(-π, π);    # Phase wrapping alternative

julia> x = [-0.5, 0.3, 1.2, 0.8];

julia> prox_state = ProximalOperators.init(unit_clamp, x);

julia> ProximalOperators.apply!(unit_clamp, prox_state, x);

julia> x  # Values clamped to [0,1]
4-element Vector{Float64}:
 0.0
 0.3
 1.0
 0.8
```

See also: [`PositiveProx`](@ref), [`ProxRule`](@ref), [`PointwiseProx`](@ref)
"""
function ClampProx(lo::Real, hi::Real)
    PointwiseProx(x -> clamp(x, lo, hi))
end
