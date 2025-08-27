function ista(p, c)
    function shrink(x::T) where {T}
        s = T(p)
        xc = x - T(c)
        if abs(xc) <= s
            return T(c)
        elseif xc > s
            return x - s
        else
            return x + s
        end
    end
    return shrink
end

struct IstaProx{F} <: StatelessProximalOperator
    f::F
    function IstaProx(s::Real, c::Real = 0)
        f = ista(s, c)
        new{typeof(f)}(f)
    end
end

function get_prox_fun(prox::IstaProx)
    prox.f
end
