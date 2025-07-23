module Modes

using ..GridUtils

abstract type Mode{N} end

function eval_mode(m::Mode{1}, x)
    error("Not implemented for $(typeof(m))")
end

function eval_mode(m::Mode{2}, x, y)
    error("Not implemented for $(typeof(m))")
end

function (m::Mode{1})(x_vec::AbstractVector; xc = 0.0)
    if !iszero(xc)
        x_vec = x_vec .+ xc
    end
    [eval_mode(m, x) for x in x_vec]
end

function (m::Mode{2})(
        x_vec::AbstractVector, y_vec::AbstractVector; θ = 0.0, xc = 0.0, yc = 0.0)
    if !iszero(xc)
        x_vec = x_vec .+ xc
    end
    if !iszero(yc)
        y_vec = y_vec .+ yc
    end
    if !iszero(θ)
        x_vec, y_vec = rotate_vectors(x_vec, y_vec; θ = θ, xc = xc, yc = yc)
    end
    [eval_mode(m, x, y) for x in x_vec, y in y_vec]
end

# function (m::Mode{2})(x_vec::AbstractVector, y_vec::AbstractVector; θ=0.0, xc=0.0, yc=0.0)
#     # Applique translation et rotation autour du centre donné
#     x_vec, y_vec = rotate_vectors(x_vec .- xc, y_vec .- yc; θ=θ)
#     x_vec = x_vec .+ xc
#     y_vec = y_vec .+ yc

#     # Évaluation sur la grille
#     [eval_mode(m, x, y) for y in y_vec, x in x_vec]  # dimensions : (ny, nx)
# end

include("measure.jl")
export intensity, phase, rms_error, correlation

include("gaussian_modes.jl")
export Gaussian1D, Gaussian, HermiteGaussian1D, HermiteGaussian
export gaussian, hermite_gaussian
export triangle_positions, gaussian_modes, hermite_gaussian_modes

end
