module FluxOptics

__precompile__()

using Requires
using LinearAlgebra

Base.copyto!(::Nothing, u) = nothing
Base.getindex(::Iterators.Cycle{Nothing}, ::Integer) = nothing
Base.lastindex(::Iterators.Cycle{Nothing}) = 1
Base.iterate(::Nothing) = (nothing, nothing)
Base.iterate(::Nothing, ::Nothing) = nothing
Iterators.reverse(::Iterators.Cycle{Nothing}) = Iterators.cycle(nothing)

isbroadcastable(a, b) = all(((m, n),) -> m == n || m == 1 || n == 1, zip(size(a), size(b)))

include("measure.jl")
export vec2D
export intensity, intensity2D, phase, rms_error, correlation

include("Fields.jl")
using .Fields
export ScalarField
export power, normalize_power!

include("GridUtils.jl")
using .GridUtils
export spatial_vectors
export Shift2D, Rot2D, Id2D

include("modes/Modes.jl")
using .Modes
export Gaussian1D, Gaussian, HermiteGaussian1D, HermiteGaussian, LaguerreGaussian
export hermite_gaussian_groups
export PointLayout, GridLayout, TriangleLayout, CustomLayout
export generate_speckle, generate_mode_stack

include("FFTutils.jl")
using .FFTutils

include("optical_components/OpticalComponents.jl")
using .OpticalComponents
export Forward, Backward
export propagate!, propagate
export AbstractOpticalComponent, AbstractOpticalSource
export AbstractCustomComponent, AbstractCustomSource
export AbstractPureComponent, AbstractPureSource
export ASProp, ASPropZ, TiltedASProp, RSProp, CollinsProp, FourierLens, ParaxialProp
export AS_BPM, TiltedAS_BPM
export ScalarSource, get_source, Phase, Mask, FourierMask, FourierPhase
export TeaDOE, TeaReflector
export FieldProbe
export BasisProjectionWrapper, make_spatial_basis, make_fourier_basis
export get_wrapped_data
export GainSheet
export OpticalChain

include("proximal_operators/ProximalOperators.jl")
using .ProximalOperators
export PointwiseProx, IstaProx, ClampProx, PositiveProx

include("optimisers_ext.jl")
using Optimisers: setup, update!, Descent, Momentum, Nesterov
export setup, update!
export make_rules, ProxRule, Fista, NoDescent, Descent, Momentum, Nesterov

using .OpticalComponents: Buffering, Buffered, Unbuffered
using .OpticalComponents: Trainability, Trainable, Static
using .OpticalComponents: propagate_and_save!, propagate_and_save
using .OpticalComponents: backpropagate!, backpropagate
using .OpticalComponents: backpropagate_with_gradient!, backpropagate_with_gradient
using .OpticalComponents: get_preallocated_gradient, alloc_gradient
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
