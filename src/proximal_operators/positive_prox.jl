function force_positive(x::T) where {T}
    if x < T(0)
        T(0)
    else
        x
    end
end

function PositiveProx()
    PointwiseProx(force_positive)
end
