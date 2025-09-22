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
        u = map(x -> similar(x.data), u0)
        v_tmp = map(x -> similar(x), v)
        c = map(((x, d),) -> similar(x, d), zip(v, dims))
        U = typeof(u)
        V = typeof(v)
        A = typeof(c)
        new{U, V, A}(u, v, v_tmp, c)
    end
end

function compute_metric(m::SquaredIntensityDifference, u::NTuple{N, ScalarField}) where {N}
    foreach(((z, x),) -> sum!(abs2, z, x.data), zip(m.v_tmp, u))
    foreach(((z, x, y),) -> (@. z = x - y), zip(m.u, m.v_tmp, m.v))
    foreach(((c, x),) -> sum!(abs2, c, x), zip(m.c, m.u))
    m.c
end

function backpropagate_metric(m::SquaredIntensityDifference,
        u::NTuple{N, ScalarField}, ∂c) where {N}
    foreach(((x, y, c),) -> (@. x *= 4*c*y.data), zip(m.u, u, ∂c))
    Tuple(map(((x, y),) -> set_field_data(x, y), zip(u, m.u)))
end
