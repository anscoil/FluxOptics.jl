function intensity(u::AbstractArray)
    n = ndims(u)
    if n <= 2
        abs.(u) .^ 2
    else
        nx, ny = size(u)
        ur = reshape(u, (nx, ny, div(length(u), nx*ny)))
        @views sum(abs.(ur) .^ 2, dims = 3)[:, :, 1]
    end
end

function rms_error(u, v)
    return sqrt(mean(abs.(u .- v) .^ 2))
end

function correlation(u, v)
    abs(dot(u, v)/(norm(u)*norm(v)))^2
end
