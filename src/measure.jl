using Statistics

function vec2D(u::AbstractArray)
    @assert ndims(u) >= 2
    reshape(eachslice(u; dims = Tuple(3:ndims(u))), :)
end

function intensity(x::T) where {T <: Number}
    abs2(x)
end

function phase(x::T) where {T <: Number}
    angle(x)
end

function intensity(u::AbstractArray)
    intensity.(u)
end

function intensity2D(u::AbstractArray)
    @assert ndims(u) >= 2
    @views sum(intensity, u, dims = Tuple(3:ndims(u)))[:, :, 1]
end

function phase(u::AbstractArray)
    phase.(u)
end

function rms_error(u, v)
    return sqrt(mean(abs2.(u .- v)))
end

function correlation(u, v)
    abs2(dot(u, v)/(norm(u)*norm(v)))
end
