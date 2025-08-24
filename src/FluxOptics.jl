module FluxOptics

__precompile__()

using Requires
using LinearAlgebra

include("Fields.jl")
using .Fields
export ScalarField
export get_data

include("measure.jl")
export vec_array2D
export intensity, intensity2D, phase, rms_error, correlation

include("GridUtils.jl")
using .GridUtils
export spatial_vectors
export Shift2D, Rot2D, Id2D

include("modes/Modes.jl")
using .Modes
export Gaussian1D, Gaussian, HermiteGaussian1D, HermiteGaussian, LaguerreGaussian
export hermite_gaussian_groups
export PointLayout, GridLayout, TriangleLayout, CustomLayout
export generate_mode_stack

include("FFTutils.jl")
using .FFTutils

include("optical_components/OpticalComponents.jl")
using .OpticalComponents
export Forward, Backward
export propagate!, propagate
export propagate_and_save!, propagate_and_save
export backpropagate!, backpropagate
export backpropagate_with_gradient!, backpropagate_with_gradient
export ASProp, RSProp, FourierLens, CollinsProp, ParaxialProp
export Phase, ScalarSource

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
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        include("plotting/Plotting.jl")
        using .Plotting
        export plot_fields, plot_fields_slider
    end
end

end
