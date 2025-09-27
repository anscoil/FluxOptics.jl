"""
    SquaredFieldDifference(fields...)

Compute squared L² norm difference between complex optical fields.

This metric calculates ‖u - v‖², the squared field difference, commonly used
as a loss function for direct field matching in inverse design problems.

# Arguments
- `fields...`: Reference ScalarField(s) to match against.

# Mathematical definition
L = ∫∫ |u(x,y) - v(x,y)|² dx dy = ‖u - v‖²

# Examples
```jldoctest
# Small example for documentation - use larger arrays in practice
julia> data1 = ones(ComplexF64, 4, 4);

julia> target = ScalarField(data1, (1.0, 1.0), 1.064);

julia> data2 = zeros(ComplexF64, 4, 4);

julia> current = ScalarField(data2, (1.0, 1.0), 1.064);

julia> metric = SquaredFieldDifference(target);

julia> metric(current)
1×1 Array{Float64, 2}:
 16.0
```

See also: `SquaredIntensityDifference`
"""
struct SquaredFieldDifference{U, V, A} <: AbstractMetric
    u::U
    v::V
    c::A

    function SquaredFieldDifference(v::Vararg{ScalarField})
        u = map(x -> similar(x.electric), v)
        c = map(x -> similar(x.electric, real(eltype(x)), extra_dims(x)), v)
        U = typeof(u)
        V = typeof(v)
        A = typeof(c)
        new{U, V, A}(u, v, c)
    end
end

function compute_metric(m::SquaredFieldDifference, u::NTuple{N, ScalarField}) where {N}
    foreach(((z, x, y),) -> (@. z = x.electric - y.electric), zip(m.u, u, m.v))
    foreach(((c, x),) -> sum!(abs2, c, x), zip(m.c, m.u))
    m.c
end

function backpropagate_metric(m::SquaredFieldDifference,
                              u::NTuple{N, ScalarField},
                              ∂c) where {N}
    foreach(((x, c),) -> (@. x *= 2*c), zip(m.u, ∂c))
    Tuple(map(((x, y),) -> set_field_data(x, y), zip(u, m.u)))
end
