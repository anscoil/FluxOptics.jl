struct PowerCoupling{M} <: AbstractMetric
    m::M

    function PowerCoupling(v::Vararg{ScalarField}; mode_selective::Bool = true)
        m = DotProduct(v...; mode_selective)
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
