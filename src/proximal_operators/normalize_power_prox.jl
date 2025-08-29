struct NormalizePowerProx{F} <: AbstractProximalOperator
    f::F
    function NormalizePowerProx(ds::NTuple, v = 1)
        function f(x)
            normalize_power!(x, ds, v)
        end
        new{typeof(f)}(f)
    end

    function NormalizePowerProx(u::ScalarField)
        v = power(u)
        f = x -> normalize_power!(x, u.ds, v)
        new{typeof(f)}(f)
    end
end

function get_prox_fun(prox::NormalizePowerProx)
    prox.f
end
