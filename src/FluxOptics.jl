module FluxOptics

using Requires
using LinearAlgebra

export Modes
export OpticalComponents

include("Types.jl")
using .Types
export Forward, Backward
export Static, Trainable

include("modes/Modes.jl")
using .Modes
export gaussian, hermite_gaussian
export triangle_positions, gaussian_modes, hermite_gaussian_modes
export intensity, rms_error, correlation
export spatial_vectors

include("optical_components/OpticalComponents.jl")
using .OpticalComponents
export ASProp, RSProp
export propagate!
export Phase

include("optimisers_ext.jl")
export rules_dict, ProxRule, Fista

include("flux_ext.jl")
include("chainrulescore_ext.jl")

end
