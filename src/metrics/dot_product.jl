struct DotProduct{V, A} <: AbstractMetric
    u::V
    v::V
    c::A
    mode_selective::Bool

    function DotProduct(v::Vararg{ScalarField}; mode_selective::Bool = true)
        u = map(x -> similar(x), v)
        if mode_selective
            c = map(x -> similar(x.data, extra_dims(x)), v)
        else
            c = map(x -> begin
                    n = prod(extra_dims(x));
                    similar(x.data, (n, n))
                end, v)
        end
        V = typeof(v)
        A = typeof(c)
        new{V, A}(u, v, c, mode_selective)
    end
end

function compute_metric(m::DotProduct, u::NTuple{N, ScalarField}) where {N}
    if m.mode_selective
        foreach(((x, y),) -> copyto!(x, y), zip(m.u, u))
        foreach(((x, y),) -> (@. x.data *= conj(y.data)), zip(m.u, m.v))
        foreach(((c, x),) -> sum!(c, x.data), zip(m.c, m.u))
    else
        foreach(
            ((x, y, c),) -> begin
                s = split_size(x)
                mul!(c, reshape(y.data, s)', reshape(x.data, s))
            end,
            zip(u, m.v, m.c))
    end
    c = if length(m.c) == 1
        m.c[1]
    else
        m.c
    end
    c
end

function backpropagate_metric(m::DotProduct, u::NTuple{N, ScalarField}, ∂c) where {N}
    foreach(((x, y),) -> copyto!(x, y), zip(m.u, m.v))
    if m.mode_selective
        foreach(((x, c),) -> (@. x.data *= c), zip(m.u, ∂c))
    else
        foreach(
            ((x, c),) -> begin
                s = split_size(x)
                mul!(reshape(x.data, s), reshape(x.data, s), c)
            end, zip(m.u, ∂c))
    end
    m.u
end
