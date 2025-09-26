"""
    SquaredIntensityDifference(field_intensity_pairs...)

Compute squared difference between field intensities and target patterns.

This metric supports two modes depending on target array dimensions:
- **Full dimensions**: Target has same dimensions as field → compare each mode individually
- **Spatial only**: Target has only spatial dimensions → compare total intensity across all modes

# Arguments
- `field_intensity_pairs...`: Tuples of (ScalarField, target_intensity_array).

# Mathematical definition  
**Full dimensions case:** L = ∫∫ |uₖ(x,y)|² - Itargetₖ(x,y)|² dx dy (per mode)

**Spatial only case:** L = ∫∫ |(∑ₖ |uₖ(x,y)|²) - Itarget(x,y)|² dx dy (total intensity)

# Examples

**Full dimensions case (mode-by-mode comparison):**
```jldoctest
# Small example for documentation - use larger arrays in practice
julia> data = ones(ComplexF64, 4, 4, 2);

julia> field = ScalarField(data, (1.0, 1.0), 1.064);

julia> target_intensity = zeros(4, 4, 2);  # Same dimensions as field

julia> metric = SquaredIntensityDifference((field, target_intensity));

julia> metric(field)
1×1×2 Array{Float64, 3}:
[:, :, 1] =
 16.0

[:, :, 2] =
 16.0
```

**Spatial only case (total intensity comparison):**
```jldoctest
# Small example for documentation - use larger arrays in practice
julia> data = ones(ComplexF64, 4, 4, 3);

julia> field = ScalarField(data, (1.0, 1.0), 1.064);

julia> target_intensity = 3*ones(4, 4);  # Only spatial dimensions

julia> metric = SquaredIntensityDifference((field, target_intensity));

julia> metric(field)
1×1 Matrix{Float64}:
 0.0
```

See also: `SquaredFieldDifference`
"""
struct SquaredIntensityDifference{U, V, A} <: AbstractMetric
    u::U
    v::V
    v_tmp::V
    c::A

    function SquaredIntensityDifference(v::Vararg{Tuple{
            ScalarField, AbstractArray{<:Real}}})
        u0, v = zip(v...)
        dims = map(
            ((x, y),) -> ntuple(k -> k <= ndims(y, true) ? 1 : size(x, k), ndims(x)),
            zip(v, u0))
        u = map(x -> similar(x.electric), u0)
        v_tmp = map(x -> similar(x), v)
        c = map(((x, d),) -> similar(x, d), zip(v, dims))
        U = typeof(u)
        V = typeof(v)
        A = typeof(c)
        new{U, V, A}(u, v, v_tmp, c)
    end
end

function compute_metric(m::SquaredIntensityDifference, u::NTuple{N, ScalarField}) where {N}
    foreach(((z, x),) -> sum!(abs2, z, x.electric), zip(m.v_tmp, u))
    foreach(((z, x, y),) -> (@. z = x - y), zip(m.u, m.v_tmp, m.v))
    foreach(((c, x),) -> sum!(abs2, c, x), zip(m.c, m.u))
    m.c
end

function backpropagate_metric(m::SquaredIntensityDifference,
        u::NTuple{N, ScalarField}, ∂c) where {N}
    foreach(((x, y, c),) -> (@. x *= 4*c*y.electric), zip(m.u, u, ∂c))
    Tuple(map(((x, y),) -> set_field_data(x, y), zip(u, m.u)))
end
