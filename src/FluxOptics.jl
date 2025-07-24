module FluxOptics

using Requires
using LinearAlgebra

export Modes
export OpticalComponents

include("GridUtils.jl")
using .GridUtils
export spatial_vectors
export Shift2D, Rot2D, Id2D

include("modes/Modes.jl")
using .Modes
export intensity, phase, rms_error, correlation
export Gaussian1D, Gaussian, HermiteGaussian1D, HermiteGaussian
export hermite_gaussian_groups
export Layout2D, triangle_layout, generate_mode_stack

include("optical_components/OpticalComponents.jl")
using .OpticalComponents
export propagate, propagate!, backpropagate, backpropagate!
export compute_gradient, compute_gradient!
export ASProp, RSProp, Phase, Seeder

include("optimisers_ext.jl")
export rules_dict, ProxRule, Fista

include("flux_ext.jl")
export OpticalChain

include("chainrulescore_ext.jl")

function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
        include("CUDAExt.jl")
        using .CUDAExt
    end
end

end
