struct DotProduct{U, V, A} <: AbstractMetric
    u::U
    v::V
    c::A
    mode_selective::Bool

    function DotProduct(v::Vararg{ScalarField}; mode_selective::Bool = true)
        u = map(x -> similar(x.data), v)
        if mode_selective
            c = map(x -> similar(x.data, extra_dims(x)), v)
        else
            c = map(x -> begin
                    n = prod(extra_dims(x));
                    similar(x.data, (n, n))
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
        foreach(((x, y),) -> copyto!(x, y.data), zip(m.u, u))
        foreach(((x, y),) -> (@. x *= conj(y.data)), zip(m.u, m.v))
        foreach(((c, x),) -> sum!(c, x), zip(m.c, m.u))
    else
        foreach(
            ((x, y, c),) -> begin
                s = split_size(x)
                mul!(c, reshape(y.data, s)', reshape(x.data, s))
            end,
            zip(u, m.v, m.c))
    end
    m.c
end

function backpropagate_metric(m::DotProduct, u::NTuple{N, ScalarField}, ∂c) where {N}
    foreach(((x, y),) -> copyto!(x, y.data), zip(m.u, m.v))
    if m.mode_selective
        foreach(((x, c),) -> (@. x *= c), zip(m.u, ∂c))
    else
        foreach(
            ((x, y, c),) -> begin
                s = split_size(y)
                mul!(reshape(x, s), reshape(x, s), c)
            end, zip(m.u, u, ∂c))
    end
    Tuple(map(((x, y),) -> set_field_data(x, y), zip(u, m.u)))
end
