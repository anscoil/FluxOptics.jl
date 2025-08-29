using Statistics

function array3D(u::AbstractArray)
    @assert ndims(u) >= 2
    nx, ny = size(u)
    reshape(u, (nx, ny, div(length(u), nx*ny)))
end

function vec_array2D(u::AbstractArray)
    @assert ndims(u) >= 3
    ur = array3D(u)
    [@view(u[:, :, k]) for k in 1:size(ur, 3)]
end

function vec_array2D(u::ScalarField)
    vec_array2D(u.data)
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
    ur = array3D(u)
    @views sum(intensity, ur, dims = 3)[:, :, 1]
end

function phase(u::AbstractArray)
    phase.(u)
end

function rms_error(u, v)
    return sqrt(mean(abs2.(u .- v)))
end

function correlation(u, v)
    abs(dot(u, v)/(norm(u)*norm(v)))^2
end
