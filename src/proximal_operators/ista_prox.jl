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

function IstaProx(s::Real, c::Real = 0)
    PointwiseProx(ista(s, c))
end
