struct SquaredFieldDifference{U, V, A} <: AbstractMetric
    u::U
    v::V
    c::A

    function SquaredFieldDifference(v::Vararg{ScalarField})
        u = map(x -> similar(x.data), v)
        c = map(x -> similar(x.data, real(eltype(x)), extra_dims(x)), v)
        U = typeof(u)
        V = typeof(v)
        A = typeof(c)
        new{U, V, A}(u, v, c)
    end
end

function compute_metric(m::SquaredFieldDifference, u::NTuple{N, ScalarField}) where {N}
    foreach(((z, x, y),) -> (@. z = x.data - y.data), zip(m.u, u, m.v))
    foreach(((c, x),) -> sum!(abs2, c, x), zip(m.c, m.u))
    m.c
end

function backpropagate_metric(m::SquaredFieldDifference,
        u::NTuple{N, ScalarField}, ∂c) where {N}
    foreach(((x, c),) -> (@. x *= 2*c), zip(m.u, ∂c))
    Tuple(map(((x, y),) -> set_field_data(x, y), zip(u, m.u)))
end
