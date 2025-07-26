using Statistics

function intensity(x::T) where {T <: Number}
    abs(x)^2
end

function phase(x::T) where {T <: Number}
    angle(x)
end

function intensity(u::AbstractArray)
    n = ndims(u)
    if n <= 2
        intensity.(u)
    else
        nx, ny = size(u)
        ur = reshape(u, (nx, ny, div(length(u), nx*ny)))
        @views sum(intensity.(ur), dims = 3)[:, :, 1]
    end
end

function phase(u::AbstractArray{T, 2}) where {T}
    phase.(u)
end

function rms_error(u, v)
    return sqrt(mean(abs.(u .- v) .^ 2))
end

function correlation(u, v)
    abs(dot(u, v)/(norm(u)*norm(v)))^2
end
