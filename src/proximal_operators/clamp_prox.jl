function ClampProx(lo::Real, hi::Real)
    PointwiseProx(x -> clamp(x, lo, hi))
end
