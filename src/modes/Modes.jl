module Modes

export spatial_vectors

function spatial_vectors(nx, ny, dx, dy; xc = 0.0, yc = 0.0)
    x_vec = ((0:(nx - 1)) .- (nx-1)/2)*dx .+ xc
    y_vec = ((0:(ny - 1)) .- (ny-1)/2)*dy .+ yc
    (x_vec, y_vec)
end

include("measure.jl")
export intensity, rms_error, correlation

include("gaussian_modes.jl")
export gaussian, hermite_gaussian
export triangle_positions, gaussian_modes, hermite_gaussian_modes

end
