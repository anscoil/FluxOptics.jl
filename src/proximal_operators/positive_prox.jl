function force_positive(x::T) where {T}
    if x < T(0)
        T(0)
    else
        x
    end
end

"""
    PositiveProx()

Project values onto the non-negative orthant (set negative values to zero).

Simple constraint for physical parameters that must be non-negative, such as
intensities, thicknesses, or absorption coefficients.

# Examples
```jldoctest
julia> pos_prox = PositiveProx();

julia> x = [-0.1, 0.0, 0.5, -0.3, 1.2];

julia> prox_state = ProximalOperators.init(pos_prox, x);

julia> ProximalOperators.apply!(pos_prox, prox_state, x);

julia> x  # Negative values set to zero
5-element Vector{Float64}:
 0.0
 0.0
 0.5
 0.0
 1.2
```

See also: [`ClampProx`](@ref), [`ProxRule`](@ref), [`PointwiseProx`](@ref)
"""
function PositiveProx()
    PointwiseProx(force_positive)
end
