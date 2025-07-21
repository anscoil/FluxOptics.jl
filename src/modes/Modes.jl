module Modes

using ..GridUtils

include("measure.jl")
export intensity, rms_error, correlation

include("gaussian_modes.jl")
export gaussian, hermite_gaussian
export triangle_positions, gaussian_modes, hermite_gaussian_modes

end
