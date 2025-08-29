struct ClampProx{F} <: PointwiseProximalOperator
    f::F
    function ClampProx(lo::Real, hi::Real)
        f = x -> clamp(x, lo, hi)
        new{typeof(f)}(f)
    end
end

function get_prox_fun(prox::ClampProx)
    prox.f
end
