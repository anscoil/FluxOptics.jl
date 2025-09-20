struct PowerCoupling{M} <: AbstractMetric
    m::M

    function PowerCoupling(v::Vararg{ScalarField}; mode_selective::Bool = true)
        m = DotProduct(v...; mode_selective)
        M = typeof(m)
        new{M}(m)
    end
end

function compute_metric(m::PowerCoupling, u::NTuple{N, ScalarField}) where {N}
    abs2.(compute_metric(m.m, u))
end

function backpropagate_metric(m::PowerCoupling, u::NTuple{N, ScalarField}, ∂c) where {N}
    ∂c = map(((c, y),) -> begin
            z = similar(y);
            copyto!(z, 2*c);
            (@. y *= z)
        end, zip(∂c, m.m.c))
    backpropagate_metric(m.m, u, ∂c)
end
