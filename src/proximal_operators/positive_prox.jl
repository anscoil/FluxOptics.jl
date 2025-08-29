function force_positive(x::T) where {T}
    if x < T(0)
        T(0)
    else
        x
    end
end

struct PositiveProx{F} <: PointwiseProximalOperator
    f::F
    function PositiveProx()
        new{typeof(force_positive)}(force_positive)
    end
end

function get_prox_fun(prox::PositiveProx)
    prox.f
end
