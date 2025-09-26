"""
    PowerCoupling(fields...; mode_selective::Bool=true)

Compute power coupled into target optical modes.

This metric calculates |⟨u,v̂⟩|², the optical power (in Watts) coupled from field u 
into normalized target modes v̂. The target fields are copied and normalized internally 
to unit power, leaving the original fields unmodified.

# Arguments
- `fields...`: Target ScalarField(s) to couple into (copied and normalized internally).
- `mode_selective::Bool=true`: If true, compute per-mode coupling. If false, compute full coupling matrix.

# Mathematical definition
P_coupled = |⟨u,v̂⟩|² where v̂ are normalized copies of the target fields

# Examples
```jldoctest
# Small example for documentation - use larger arrays in practice
julia> data = ones(ComplexF64, 4, 4, 2);

julia> data[3:end,:,2] .= -ones(ComplexF64, 2, 4);

julia> u = ScalarField(data, (1.0, 1.0), 1.064);

julia> normalize_power!(u);

julia> metric_selective = PowerCoupling(u);

julia> metric_selective(u)
1×1×2 Array{Float64, 3}:
[:, :, 1] =
 1.0

[:, :, 2] =
 1.0

julia> metric_non_selective = PowerCoupling(u; mode_selective = false);

julia> metric_non_selective(u)
2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0
```

See also: `DotProduct`
"""
struct PowerCoupling{M} <: AbstractMetric
    m::M

    function PowerCoupling(v::Vararg{ScalarField}; mode_selective::Bool = true)
        v_nrm = map(x -> normalize_power!(copy(x)), v)
        m = DotProduct(v_nrm...; mode_selective)
        M = typeof(m)
        new{M}(m)
    end
end

function compute_metric(m::PowerCoupling, u::NTuple{N, ScalarField}) where {N}
    map(x -> abs2.(x), compute_metric(m.m, u))
end

function backpropagate_metric(m::PowerCoupling, u::NTuple{N, ScalarField}, ∂c) where {N}
    ∂c = map(((c, y),) -> (@. y *= 2*c), zip(∂c, m.m.c))
    backpropagate_metric(m.m, u, ∂c)
end
