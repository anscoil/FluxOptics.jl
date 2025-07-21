module GridUtils

export spatial_vectors

function spatial_vectors(nx, ny, dx, dy; xc = 0.0, yc = 0.0)
    x_vec = ((0:(nx - 1)) .- (nx-1)/2)*dx .+ xc
    y_vec = ((0:(ny - 1)) .- (ny-1)/2)*dy .+ yc
    (x_vec, y_vec)
end

end
