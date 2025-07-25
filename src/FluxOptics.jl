module FluxOptics

__precompile__()

using Requires
using LinearAlgebra

include("GridUtils.jl")
using .GridUtils
export spatial_vectors
export Shift2D, Rot2D, Id2D

include("modes/Modes.jl")
using .Modes
export intensity, phase, rms_error, correlation
export Gaussian1D, Gaussian, HermiteGaussian1D, HermiteGaussian, LaguerreGaussian
export hermite_gaussian_groups
export PointLayout, GridLayout, TriangleLayout, CustomLayout
export generate_mode_stack

include("optical_components/OpticalComponents.jl")
using .OpticalComponents
export propagate!, propagate
export propagate_and_save!, propagate_and_save
export backpropagate!, backpropagate
export backpropagate_with_gradient!, backpropagate_with_gradient
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
