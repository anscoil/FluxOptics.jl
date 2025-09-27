# FluxOptics.jl - Inverse Optics Design Library  
# Copyright (c) 2025 Nicolas Barré
# MIT License - see LICENSE file for details

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
bzip(x...) = Base.broadcasted(tuple, x...)

function Base.similar(A::Type{<: AbstractArray}, ndims::Integer)
    @assert isconcretetype(A)
    A.name.wrapper{A.parameters[1], ndims, A.parameters[3:end]...}
end

function Base.similar(A::Type{<: AbstractArray}, f::Function)
    @assert isconcretetype(A)
    A.name.wrapper{f(A.parameters[1]), A.parameters[2:end]...}
end

function Base.similar(A::Type{<: AbstractArray}, f::Function, ndims::Integer)
    @assert isconcretetype(A)
    A.name.wrapper{f(A.parameters[1]), ndims, A.parameters[3:end]...}
end

include("Fields.jl")
using .Fields
export ScalarField
export set_field_data
export dot, power, normalize_power!, coupling_efficiency, intensity, phase

include("metrics/Metrics.jl")
using .Metrics
export DotProduct, PowerCoupling, SquaredFieldDifference, SquaredIntensityDifference

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
export AbstractPipeComponent, AbstractOpticalSource
export AbstractCustomComponent, AbstractCustomSource
export AbstractPureComponent, AbstractPureSource
export ASProp, ASPropZ, TiltedASProp, ShiftProp
export RSProp, CollinsProp, FourierLens, ParaxialProp
export as_rotation!, as_rotation, plan_as_rotation, field_rotation_matrix
export AS_BPM, TiltedAS_BPM, Shift_BPM
export ScalarSource, get_source, Phase, Mask, FourierMask, FourierPhase
export TeaDOE, TeaReflector
export FieldProbe
export BasisProjectionWrapper, make_spatial_basis, make_fourier_basis
export GainSheet
export OpticalSequence, FourierWrapper
export OpticalSystem, get_source, get_components
export get_data

include("OptimisersExt.jl")
using .OptimisersExt
import Optimisers: setup, update!, Descent, Momentum, Nesterov
export make_rules, setup, update!
export ProxRule, Descent, Momentum, Nesterov, Fista, NoDescent
export PointwiseProx, IstaProx, ClampProx, PositiveProx, TVProx
export TV_denoise!
export ProximalOperators

include("ChainRulesCoreExt.jl")

# include("cuda/CUDAExt.jl")
# using .CUDAExt

function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
        include("cuda/CUDAExt.jl")
        using .CUDAExt
    end
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        include("plotting/Plotting.jl")
        using .Plotting
        export visualize, visualize_slider
    end
end

end
