module FluxOptics

__precompile__()

using Requires
using LinearAlgebra

include("Fields.jl")
using .Fields
export ScalarField
export get_data, collect_data
export power, normalize_power!

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
export init!, propagate!, propagate
export propagate_and_save!, propagate_and_save
export backpropagate!, backpropagate
export backpropagate_with_gradient!, backpropagate_with_gradient
export ASProp, ASPropZ, RSProp, CollinsProp, FourierLens, ParaxialProp
export ScalarSource, get_source, Phase, ComplexMask, TeaDOE, TeaReflector
export FieldProbe

include("proximal_operators/ProximalOperators.jl")
using .ProximalOperators
export PointwiseProx, IstaProx, ClampProx, PositiveProx

include("optimisers_ext.jl")
export rules_dict, ProxRule, Fista

include("flux_ext.jl")
export OpticalChain, set_kwargs!

include("chainrulescore_ext.jl")

function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
        include("CUDAExt.jl")
        using .CUDAExt
    end
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        include("plotting/Plotting.jl")
        using .Plotting
        export visualize, visualize_slider
    end
end

end
