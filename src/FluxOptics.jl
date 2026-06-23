# FluxOptics.jl - Inverse Optics Design Library  
# Copyright (c) 2025 Nicolas Barré
# MIT License - see LICENSE file for details

module FluxOptics

__precompile__()

using LinearAlgebra

Base.copyto!(::Nothing, u) = nothing
Base.getindex(x::Real, ::CartesianIndex{0}) = x
Base.getindex(x::Real, ::CartesianIndex) = x

struct NothingIterator end
Base.iterate(::NothingIterator) = (nothing, nothing)
Base.iterate(::NothingIterator, ::Any) = (nothing, nothing)
Base.getindex(::NothingIterator, ::Any) = nothing
Base.lastindex(::NothingIterator) = 1
Iterators.reverse(::NothingIterator) = NothingIterator()

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

Base.similar(t::Tuple{}) = ()
Base.similar(t::NamedTuple{}) = (;)

_get_val(A::AbstractArray{<:Any, 2}, I, J) = A[I]
function _get_val(A::AbstractArray{<:Any, N}, I, J) where {N}
    A[I, CartesianIndex(min.(Tuple(J), size(A)[3:end]))]
end

include("fields/Fields.jl")
using .Fields
export ScalarField, ScalarWaveField
export set_field_data, set_field_tilts, offset_tilts!, is_on_axis
export dot, power, normalize_power!, coupling_efficiency, intensity, phase
export orthonormalize, unitary_transform, spatial_moments, spatial_centroids, spatial_variance
export split_field, poynting_flux, normalize_poynting!

include("metrics/Metrics.jl")
using .Metrics
export Metrics
export AbstractMetric
export DotProduct, PowerCoupling, SquaredFieldDifference, SquaredIntensityDifference

include("GridUtils.jl")
using .GridUtils
export spatial_vectors
export Shift2D, Rot2D, Id2D

include("modes/Modes.jl")
export Modes
using .Modes
export Gaussian1D, Gaussian, HermiteGaussian1D, HermiteGaussian, LaguerreGaussian
export hermite_gaussian_groups
export generate_speckle, generate_mode_stack

include("FFTutils.jl")
using .FFTutils

include("optical_components/OpticalComponents.jl")
using .OpticalComponents
export OpticalComponents
export Trainability, Trainable, Static, Buffered, Unbuffered
export trainable, istrainable, isbuffered
export propagate!, propagate
export AbstractOpticalComponent, AbstractPipeComponent, AbstractOpticalSource
export AbstractCustomComponent, AbstractCustomSource
export AbstractPureComponent, AbstractPureSource
export pad, crop, PadCropOperator
export TiltAnchor, ASProp, ASPropZ, ShiftProp
export RSProp, CollinsProp, FourierLens, ParaxialProp
export as_rotation!, as_rotation, field_rotation_matrix
export AS_BPM, Shift_BPM
export FS_WPM, smoothstep_partition
export ScalarSource, get_source, Phase, Mask, FourierMask, FourierPhase
export TeaDOE, TeaReflector
export FieldProbe
export BasisProjectionWrapper, make_spatial_basis, make_fourier_basis
export FourierSmoothingWrapper, DensityWrapper, set_sharpness!, set_binarize!
export GainSheet
export AbstractSequence, OpticalSequence, FourierOperator, FourierWrapper, get_sequence
export OpticalSystem, get_components
export get_data

export ScalarWaveSource
export ScalarWavePropagator, ScalarWaveBPM
export BidirectionalSystem, GmresSolver, fp_solve!
export BicgstabSolver, BilqSolver, CgneSolver, CraigSolver
export test_adjoint

include("OptimisersExt.jl")
using .OptimisersExt
import Optimisers: setup, update!, Descent
export make_rules, setup, update!
export AbstractProximalOperator
export ProxRule, Fista, NoDescent, Descent
export PointwiseProx, IstaProx, ClampProx, PositiveProx, TVProx
export TV_denoise!
export ProximalOperators

include("ChainRulesCoreExt.jl")

# Plotting extension (loaded when Makie is available)
function visualize end
function visualize_slider end
export visualize, visualize_slider

end
