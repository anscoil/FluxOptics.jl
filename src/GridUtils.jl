module GridUtils

export spatial_vectors, rotate_vectors

function spatial_vectors(nx, ny, dx, dy; xc = 0.0, yc = 0.0)
    x_vec = ((0:(nx - 1)) .- (nx-1)/2)*dx .+ xc
    y_vec = ((0:(ny - 1)) .- (ny-1)/2)*dy .+ yc
    (x_vec, y_vec)
end

function rotate_vectors(x_vec::AbstractVector, y_vec::AbstractVector;
        θ = 0.0, xc = 0.0, yc = 0.0)
    X = x_vec .- xc
    Y = y_vec .- yc
    Xr = cos(θ) .* X .- sin(θ) .* Y
    Yr = sin(θ) .* X .+ cos(θ) .* Y
    return Xr .+ xc, Yr .+ yc
end

end
