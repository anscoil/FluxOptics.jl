"""
    DotProduct(fields...; mode_selective::Bool=true)

Compute dot product (inner product) between optical fields for optimization.

This metric calculates the complex overlap integral ⟨u,v⟩ between fields, commonly used 
for mode coupling analysis and field matching objectives in inverse design.

# Arguments
- `fields...`: Reference ScalarField(s) to compare against.
- `mode_selective::Bool=true`: If true, compute per-mode overlaps. If false, compute full overlap matrix.

# Mathematical definition
⟨u,v⟩ = ∫∫ u*(x,y) v(x,y) dx dy

# Examples
```jldoctest
# Small example for documentation - use larger arrays in practice
julia> data = ones(ComplexF64, 4, 4, 2);

julia> data[3:end,:,2] .= -ones(ComplexF64, 2, 4);

julia> u = ScalarField(data, (1.0, 1.0), 1.064);

julia> normalize_power!(u);

julia> metric_selective = DotProduct(u);

julia> metric_selective(u)
1×1×2 Array{ComplexF64, 3}:
[:, :, 1] =
 1.0 + 0.0im

[:, :, 2] =
 1.0 + 0.0im

julia> metric_non_selective = DotProduct(u; mode_selective = false);

julia> metric_non_selective(u)
2×2 Matrix{ComplexF64}:
 1.0+0.0im  0.0+0.0im
 0.0+0.0im  1.0+0.0im
```

See also: `PowerCoupling`
"""
struct DotProduct{U, V, A} <: AbstractMetric
    u::U
    v::V
    c::A
    mode_selective::Bool

    function DotProduct(v::Vararg{ScalarField}; mode_selective::Bool = true)
        u = map(x -> similar(x.electric), v)
        if mode_selective
            c = map(x -> similar(x.electric, extra_dims(x)), v)
        else
            c = map(x -> begin
                        n = prod(extra_dims(x));
                        similar(x.electric, (n, n))
                    end, v)
        end
        U = typeof(u)
        V = typeof(v)
        A = typeof(c)
        new{U, V, A}(u, v, c, mode_selective)
    end
end

function compute_metric(m::DotProduct, u::NTuple{N, ScalarField}) where {N}
    if m.mode_selective
        foreach(((x, y),) -> copyto!(x, y.electric), zip(m.u, u))
        foreach(((x, y),) -> begin
                    ds = prod(y.ds)
                    (@. x *= conj(y.electric)*ds)
                end, zip(m.u, m.v))
        foreach(((c, x),) -> sum!(c, x), zip(m.c, m.u))
    else
        foreach(((x, y, c),) -> begin
                    s = split_size(x)
                    mul!(c, reshape(y.electric, s)', reshape(x.electric, s))
                    c .*= prod(y.ds)
                end,
                zip(u, m.v, m.c))
    end
    m.c
end

function backpropagate_metric(m::DotProduct, u::NTuple{N, ScalarField}, ∂c) where {N}
    foreach(((x, y),) -> copyto!(x, y.electric), zip(m.u, m.v))
    if m.mode_selective
        foreach(((x, y, c),) -> (@. x *= c), zip(m.u, u, ∂c))
    else
        foreach(((x, y, c),) -> begin
                    s = split_size(y)
                    mul!(reshape(x, s), reshape(x, s), c)
                end, zip(m.u, u, ∂c))
    end
    Tuple(map(((x, y),) -> set_field_data(x, y), zip(u, m.u)))
end
