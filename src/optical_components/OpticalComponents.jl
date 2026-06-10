module OpticalComponents

using Functors
using LinearAlgebra
using KernelAbstractions
using AbstractFFTs
using FINUFFT
using EllipsisNotation
using LRUCache
using Adapt
using ..FluxOptics: isbroadcastable, bzip, NothingIterator
using ..GridUtils
using ..Fields
using ..FFTutils

export Direction, Forward, Backward
export Trainability, Trainable, Static, Buffering, Buffered, Unbuffered
export AbstractOpticalComponent, AbstractPipeComponent, AbstractOpticalSource
export AbstractCustomComponent, AbstractCustomSource
export AbstractPureComponent, AbstractPureSource
export propagate!, propagate
export get_data
export trainable, istrainable, isbuffered

abstract type Direction end

struct Forward <: Direction end

struct Backward <: Direction end

Base.reverse(::Type{Forward}) = Backward
Base.reverse(::Type{Backward}) = Forward

Base.sign(::Type{Forward}) = 1
Base.sign(::Type{Backward}) = -1

"""
    Buffering

Abstract type for gradient and buffer allocation strategy.

Controls how memory is managed for trainable components during optimization,
allowing trade-offs between performance and memory usage.

# Subtypes
- [`Buffered`](@ref): Pre-allocated buffers for maximum performance
- [`Unbuffered`](@ref): Dynamic allocation for memory efficiency

See also: [`Trainable`](@ref), [`Trainability`](@ref)
"""
abstract type Buffering end

"""
    Buffered <: Buffering

Buffering strategy with pre-allocated gradient and forward-pass buffers.

Components using `Buffered` pre-allocate all necessary buffers during construction,
providing maximum performance during optimization at the cost of higher memory usage.
Recommended for production training and repeated optimizations.

# Advantages
- **Maximum performance**: No allocation overhead during training
- **Predictable memory**: All buffers allocated upfront
- **GPU friendly**: Reduces memory fragmentation

# Examples
```jldoctest
julia> u = ScalarField(ones(ComplexF64, 64, 64), (2.0, 2.0), 1.064);

julia> mask = Mask(u, (x, y) -> 1.0; trainable=true, buffered=true);

julia> typeof(mask) <: AbstractPipeComponent{Trainable{Buffered}}
true
```

See also: [`Unbuffered`](@ref), [`Trainability`](@ref)
"""
struct Buffered <: Buffering end

"""
    Unbuffered <: Buffering

Buffering strategy with dynamic allocation of gradients and buffers.

Components using `Unbuffered` allocate gradients and buffers as needed during
optimization, providing memory efficiency at the cost of allocation overhead.
Recommended for prototyping and memory-constrained environments.

# Advantages
- **Memory efficient**: Only allocates what's needed
- **Flexible**: Easy to modify component structure
- **Debug friendly**: Clear allocation patterns

# Examples
```jldoctest
julia> u = ScalarField(ones(ComplexF64, 64, 64), (2.0, 2.0), 1.064);

julia> phase_mask = Phase(u, (x, y) -> 0.0; trainable=true, buffered=false);

julia> typeof(phase_mask) <: AbstractPipeComponent{Trainable{Unbuffered}}
true
```

See also: [`Buffered`](@ref), [`Trainability`](@ref)
"""
struct Unbuffered <: Buffering end

"""
    Trainability

Abstract type for trainability classification of optical components.

Defines whether a component's parameters can be optimized and how gradients
are managed during automatic differentiation.

# Subtypes
- [`Static`](@ref): Non-trainable component
- [`Trainable{Buffering}`](@ref): Trainable component with buffer management

See also: [`Buffering`](@ref), [`Trainability`](@ref)
"""
abstract type Trainability end

"""
    Static <: Trainability

Trainability type for non-optimizable optical components.

Components with `Static` trainability have fixed parameters that cannot be
modified during optimization. These components are more efficient as they
don't allocate gradients or maintain optimization state.

# Examples
```jldoctest
julia> u = ScalarField(ones(ComplexF64, 64, 64), (2.0, 2.0), 1.064);

julia> lens = FourierLens(u, (2.0, 2.0), 1000.0);  # Static by default

julia> typeof(lens) <: AbstractPipeComponent{Static}
true
```

See also: [`Trainable`](@ref), [`Trainability`](@ref)
"""
struct Static <: Trainability end

"""
    Trainable{B <: Buffering} <: Trainability

Trainability type for optimizable optical components.

Components with `Trainable` trainability have parameters that can be optimized
during training. The type parameter `B` controls gradient and buffer allocation
strategy for performance optimization.

# Type Parameter
- `B::Buffering`: [`Buffered`](@ref) or [`Unbuffered`](@ref)

# Examples
```jldoctest
julia> u = ScalarField(ones(ComplexF64, 64, 64), (2.0, 2.0), 1.064);

julia> phase_mask = Phase(u, (x, y) -> 0.0; trainable=true, buffered=true);

julia> typeof(phase_mask) <: AbstractPipeComponent{<:Trainable}
true
```

See also: [`Static`](@ref), [`Buffering`](@ref), [`Trainable`](@ref)
"""
struct Trainable{A <: Buffering} <: Trainability end

function trainability(trainable::Bool, buffered::Bool)
    if trainable
        if buffered
            Trainable{Buffered}
        else
            Trainable{Unbuffered}
        end
    else
        if buffered
            @warn "Invalid combination: `buffered=true` only makes sense when \
            `trainable=true`.\nIgnoring buffering."
        end
        Static
    end
end

"""
    AbstractOpticalComponent{M}

Abstract supertype for all optical components in FluxOptics.

Root type for the optical component hierarchy, encompassing both sources that
generate fields and pipe components that transform fields. The type parameter
`M` indicates the trainability and buffering strategy.

# Type Parameter
- `M::Trainability`: Component trainability (`Static`, `Trainable{Buffered}`, etc.)

# Subtypes
- [`AbstractOpticalSource`](@ref): Components that generate optical fields
- [`AbstractPipeComponent`](@ref): Components that transform optical fields

See also: [`Trainability`](@ref), [`get_data`](@ref)
"""
abstract type AbstractOpticalComponent{M <: Trainability} end

get_trainability(p::AbstractOpticalComponent{M}) where {M} = M

"""
    istrainable(component::AbstractOpticalComponent) -> Bool

Check if a component is trainable (has optimizable parameters).

Returns `true` for components created with `trainable=true`, `false` otherwise.

# Examples
```julia
phase_static = Phase(u, (x, y) -> x^2)
istrainable(phase_static)  # false

phase_train = Phase(u, (x, y) -> 0.0; trainable=true)
istrainable(phase_train)  # true
```

See also: [`isbuffered`](@ref)
"""
istrainable(p::AbstractOpticalComponent{Static}) = false
istrainable(p::AbstractOpticalComponent{<:Trainable}) = true

"""
    isbuffered(component::AbstractOpticalComponent) -> Bool

Check if a trainable component uses pre-allocated gradient buffers.

Returns `true` for components created with `trainable=true, buffered=true`,
`false` otherwise.

# Examples
```julia
phase_unbuf = Phase(u, (x, y) -> 0.0; trainable=true, buffered=false)
isbuffered(phase_unbuf)  # false

phase_buf = Phase(u, (x, y) -> 0.0; trainable=true, buffered=true)
isbuffered(phase_buf)  # true
```

See also: [`istrainable`](@ref), [`Buffered`](@ref), [`Unbuffered`](@ref)
"""
isbuffered(p::AbstractOpticalComponent) = false
isbuffered(p::AbstractOpticalComponent{Trainable{Buffered}}) = true

function data_symbol(p::AbstractOpticalComponent)
    error("Not implemented")
end

"""
    get_data(component::AbstractOpticalComponent)

Access the internal data array(s) of an optical component.

Returns the underlying parameter array (phase, mask, etc.) or tuple of arrays
for components with multiple parameters. Used for inspection and direct manipulation.

# Returns
- Single `AbstractArray` for simple components (Phase, Mask)
- `Tuple` of arrays for components with multiple parameters
- May be on GPU if component was constructed with GPU arrays

# Examples
```julia
phase = Phase(u, (x, y) -> x^2; trainable=true)
φ = get_data(phase)  # Returns phase array
```
"""
function get_data(p::AbstractOpticalComponent)
    getproperty(p, data_symbol(p))
end

function auxiliary_trainable(p::AbstractOpticalComponent)
    inner = trainable(p)
    skip = data_symbol(p)
    NamedTuple(k => v for (k, v) in pairs(inner) if k !== skip)
end

"""
    Base.collect(component::AbstractOpticalComponent)

Convert component data to CPU arrays.

Useful for transferring data from GPU to CPU for analysis or visualization.

# Examples
```julia
# GPU component
phase_gpu = Phase(u_gpu, (x, y) -> x^2)

# Transfer to CPU
φ_cpu = collect(phase_gpu)
```

See also: [`get_data`](@ref)
"""
function Base.collect(p::AbstractOpticalComponent)
    data = get_data(p)
    if isa(data, Tuple)
        map(collect, data)
    else
        collect(data)
    end
end

"""
    Base.length(::AbstractOpticalComponent) -> Int

Return the number of components. Always returns 1 for atomic components.

Used for compatibility with iteration protocols.
"""
Base.length(p::AbstractOpticalComponent) = 1

"""
    Base.size(component::AbstractOpticalComponent)

Return the size of the component's data array.

# Examples
```julia
phase = Phase(u, (x, y) -> x^2)
size(phase)  # (128, 128) if u is 128×128×...
```
"""
Base.size(p::AbstractOpticalComponent) = size(get_data(p))

"""
    Base.fill!(component::AbstractOpticalComponent, value::Number)
    Base.fill!(component::AbstractOpticalComponent, value::AbstractArray)

Fill component data with a constant value or array.

Modifies the component in-place by setting all elements of its data array(s)
to the specified value (scalar broadcast) or by copying the array.

# Arguments
- `component`: Optical component to modify
- `value`: Scalar to broadcast or array to copy

# Returns
The modified data array(s).

# Examples
```julia
phase = Phase(u, (x, y) -> randn(); trainable=true)

# Reset to zero (scalar)
fill!(phase, 0.0)

# Fill with array
new_phase = 0.01 .* (xv.^2 .+ yv'.^2)
fill!(phase, new_phase)
```
"""
function Base.fill!(p::AbstractOpticalComponent, v::Number)
    data = get_data(p)
    if isa(data, Tuple)
        foreach(data -> isa(data, AbstractArray) ? data .= v : nothing, get_data(p))
    else
        data .= v
    end
    data
end

function Base.fill!(p::AbstractOpticalComponent, v::AbstractArray)
    data = get_data(p)
    isa(data, Tuple) && error("fill! with an array is ambiguous for multi-data components")
    copyto!(data, v)
    data
end

function function_to_array(f::Function, ns::NTuple{Nd, Integer}, ds::NTuple{Nd, Real},
                           isfourier = false) where {Nd}
    if isfourier
        xs = [fftfreq(nx, 1/dx) for (nx, dx) in zip(ns, ds)]
    else
        xs = spatial_vectors(ns, ds)
    end
    Nd == 2 ? f.(xs[1], xs[2]') : f.(xs[1])
end

function Base.fill!(p::AbstractOpticalComponent, f::Function, ds::NTuple{Nd, Real};
                    isfourier=false) where {Nd}
    data = get_data(p)
    isa(data, Tuple) && error("fill! with a function is ambiguous for multi-data components")
    copyto!(data, function_to_array(f, size(data), ds, isfourier))
end

"""
    trainable(component::AbstractOpticalComponent)

Return trainable parameters of a component as a NamedTuple.

For `Static` components, returns an empty `NamedTuple{}()`.
For `Trainable` components, returns a `NamedTuple` with parameter names as keys
and parameter arrays as values.

# Returns
- `NamedTuple{}()` for static components
- `NamedTuple` with parameter arrays for trainable components

# Examples
```julia
# Static component
phase_static = Phase(u, (x, y) -> x^2)
trainable(phase_static)  # NamedTuple{}()

# Trainable component
phase_train = Phase(u, (x, y) -> 0.0; trainable=true)
params = trainable(phase_train)  # (; ϕ = ...)

# Use with optimization
using Functors
all_params = Functors.fmap(trainable, system)
```

See also: [`istrainable`](@ref)
"""
trainable(p::AbstractOpticalComponent{Static}) = NamedTuple{}()

function trainable(p::AbstractOpticalComponent{<:Trainable})
    error("Not implemented")
end

"""
    AbstractPipeComponent{M} <: AbstractOpticalComponent{M}

Abstract type for optical components that process incident fields.

Pipe components transform optical fields that pass through them (like a pipe
transforms fluid flow), as opposed to `AbstractOpticalSource` which generate 
fields from nothing. The type parameter `M` indicates trainability.

# Type Parameter
- `M::Trainability`: `Static`, `Trainable{Unbuffered}`, or `Trainable{Buffered}`

# Subtypes
- [`AbstractCustomComponent`](@ref): Stateful components with custom propagation
- [`AbstractPureComponent`](@ref): Stateless components with simple propagation

# Required Interface
All subtypes must implement:
- `propagate!(u, component)` or `propagate(u, component)`
- `get_data(component)`: Access to component parameters

See also: [`AbstractOpticalSource`](@ref), [`propagate!`](@ref)
"""
abstract type AbstractPipeComponent{M} <: AbstractOpticalComponent{M} end

"""
    AbstractPureComponent{M} <: AbstractPipeComponent{M}

Abstract type for optical components with functional interface.

Pure components provide a functional interface where the same input always gives
the same output, without requiring manual implementation of gradient rules. They
can wrap complex internal state (including `AbstractCustomComponent`s) but expose
a pure functional interface that works seamlessly with automatic differentiation.

# Type Parameter
- `M::Trainability`: Usually `Static`, but can support `Trainable` for differentiable components

# Required Methods
All subtypes must implement:
- `propagate(u, component)`: Direct field transformation
- `get_data(component)`: Access to component parameters

# Characteristics
- **Functional interface**: Same input → same output, regardless of internal complexity
- **Differentiable**: Automatic differentiation works without custom rules
- **Composable**: Can wrap and combine other components
- **Implementation agnostic**: Internal state hidden behind pure interface

# Examples
```jldoctest
julia> u = ScalarField(ones(ComplexF64, 64, 64), (2.0, 2.0), 1.064);

julia> prop_z = ASPropZ(u, 500.0; trainable=true);

julia> typeof(prop_z) <: AbstractPureComponent
true
```

See also: [`AbstractCustomComponent`](@ref), [`ASPropZ`](@ref), [`OpticalSequence`](@ref)
"""
abstract type AbstractPureComponent{M} <: AbstractPipeComponent{M} end

"""
    propagate(u::ScalarField, component::AbstractPipeComponent)
    propagate(source::AbstractOpticalSource)

Propagate an optical field through a component or generate a field from a source.

The first form creates a copy of the input field and applies the optical transformation
defined by `component`. The second form generates a new optical field from a source
component.

# Arguments
- `u::ScalarField`: Input optical field (unchanged)
- `component`: Optical component to propagate through  
- `source::AbstractOpticalSource`: Source component to generate field from

# Returns
New `ScalarField` with the transformation applied or generated.

# Examples
```jldoctest
julia> u = ScalarField(ones(ComplexF64, 64, 64), (2.0, 2.0), 1.064);

julia> phase_mask = Phase(u, (x, y) -> 0.1*(x^2 + y^2));

julia> u_prop = propagate(u, phase_mask);

julia> source = ScalarSource(u; trainable=true);

julia> u_generated = propagate(source);

julia> size(u_generated) == size(u)
true
```

See also: [`propagate!`](@ref)
"""
function propagate(u, p::AbstractPureComponent)
    error("Not implemented")
end

"""
    propagate!(u::ScalarField, component::AbstractPipeComponent)

Propagate field through component in-place, modifying the input field.

This is the core in-place propagation method that modifies the field as it passes through
the optical component. The field is transformed according to the component's optical
properties.

# Arguments
- `u::ScalarField`: Field to propagate (modified in-place)
- `component`: Optical component to propagate through

# Returns
Modified `ScalarField` (same object as input).

# Examples
```julia
u = ScalarField(ones(ComplexF64, 128, 128), (2.0, 2.0), 1.064)
phase = Phase(u, (x, y) -> 0.1*x^2)

# In-place propagation
propagate!(u, phase)  # u is modified

# For sequence of components
propagate!(u, phase1)
propagate!(u, lens)
propagate!(u, phase2)
```

See also: [`propagate`](@ref)
"""
propagate!(u, p::AbstractPureComponent) = propagate(u, p)

function backpropagate!(∂v, p::AbstractPureComponent)
    error("Not implemented")
end

backpropagate(∂v, p::AbstractPureComponent) = backpropagate!(copy(∂v), p)

"""
    AbstractCustomComponent{M} <: AbstractPipeComponent{M}

Abstract type for stateful optical components with custom propagation logic.

Custom components maintain internal state, support gradient computation for 
optimization, and implement complex propagation behavior. They are the building
blocks for trainable optical elements like phase masks, diffractive elements, etc.

# Type Parameter
- `M::Trainability`: Determines gradient and buffer management

# Required Methods
All subtypes must implement:
- `propagate!(u, component)`: In-place field transformation
- `get_data(component)`: Access to trainable parameters  
- `trainable(component)`: Return trainable parameters (if `M <: Trainable`)

# Optional Methods (for optimization)
- `backpropagate!(u, component)`: Reverse propagation
- `get_preallocated_gradient(component)`: Pre-allocated gradients (if buffered)
- `alloc_saved_buffer(u, component)`: Allocate forward-pass buffers

# Examples
```jldoctest
julia> u = ScalarField(ones(ComplexF64, 64, 64), (2.0, 2.0), 1.064);

julia> phase_mask = Phase(u, (x, y) -> 0.1*x^2; trainable=true);

julia> typeof(phase_mask) <: AbstractCustomComponent
true

julia> propagate!(u, phase_mask);
```

See also: [`AbstractPureComponent`](@ref), [`Phase`](@ref), [`Mask`](@ref)
"""
abstract type AbstractCustomComponent{M} <: AbstractPipeComponent{M} end

function alloc_gradient(p::AbstractCustomComponent{Trainable{Unbuffered}})
    map(similar, trainable(p))
end

function get_preallocated_gradient(p::AbstractCustomComponent{Trainable{Buffered}})
    error("Not implemented")
end

function alloc_saved_buffer(u, p::AbstractCustomComponent{Trainable{Unbuffered}})
    error("Not implemented")
end

function get_saved_buffer(p::AbstractCustomComponent{Trainable{Buffered}})
    error("Not implemented")
end

function _propagate!(u, p::AbstractCustomComponent, direction::Type{<:Direction})
    error("Not implemented")
end

propagate!(u, p::AbstractCustomComponent) = _propagate!(u, p, Forward)

function propagate_and_save!(u, u_saved, p::AbstractCustomComponent{<:Trainable})
    error("Not implemented")
end

function propagate_and_save!(u, p::AbstractCustomComponent{Trainable{Buffered}})
    u_saved = get_saved_buffer(p)
    propagate_and_save!(u, u_saved, p)
end

backpropagate!(∂v, p::AbstractCustomComponent) = _propagate!(∂v, p, Backward)

backpropagate(∂v, p::AbstractCustomComponent) = backpropagate!(copy(∂v), p)

function backpropagate_with_gradient!(∂v, u_saved, ∂p::NamedTuple,
                                      p::AbstractCustomComponent{<:Trainable})
    error("Not implemented")
end

function propagate(u, p::AbstractCustomComponent)
    propagate!(copy(u), p)
end

function propagate_and_save(u, p::AbstractCustomComponent{Trainable{Buffered}})
    propagate_and_save!(copy(u), p)
end

function propagate_and_save(u, u_saved, p::AbstractCustomComponent{Trainable{Unbuffered}})
    propagate_and_save!(copy(u), u_saved, p)
end

function backpropagate_with_gradient(∂v, u_saved, ∂p::NamedTuple,
                                     p::AbstractCustomComponent{<:Trainable})
    backpropagate_with_gradient!(copy(∂v), u_saved, ∂p, p)
end

"""
    AbstractOpticalSource{M} <: AbstractOpticalComponent{M}

Abstract type for optical components that generate fields.

Sources create optical fields from nothing (like a light source), as opposed to
`AbstractPipeComponent` which transform existing fields. Sources are typically
placed at the beginning of optical systems.

# Type Parameter  
- `M::Trainability`: `Static`, `Trainable{Unbuffered}`, or `Trainable{Buffered}`

# Subtypes
- [`AbstractCustomSource`](@ref): Sources with custom generation logic
- [`AbstractPureSource`](@ref): Sources with simple generation

# Required Interface
All subtypes must implement:
- `propagate(source)`: Generate the optical field
- `get_data(source)`: Access to source parameters

# Examples
```jldoctest
julia> u = ScalarField(ones(ComplexF64, 64, 64), (2.0, 2.0), 1.064);

julia> source = ScalarSource(u; trainable=true);

julia> field = propagate(source);
```

See also: [`AbstractPipeComponent`](@ref), [`ScalarSource`](@ref), [`propagate`](@ref)
"""
abstract type AbstractOpticalSource{M} <: AbstractOpticalComponent{M} end

function propagate(p::AbstractOpticalSource)
    error("Not implemented")
end

"""
    AbstractPureSource{M} <: AbstractOpticalSource{M}

Abstract type for stateless optical sources with direct generation.

Pure sources generate optical fields through direct computation without 
maintaining internal state, suitable for simple or static source configurations.

# Type Parameter
- `M::Trainability`: Usually `Static`, but can support simple trainable cases

# Required Methods
All subtypes must implement:
- `propagate(source)`: Generate the optical field directly
- `get_data(source)`: Access to source parameters

See also: [`AbstractCustomSource`](@ref), [`ScalarSource`](@ref)
"""
abstract type AbstractPureSource{M} <: AbstractOpticalSource{M} end

"""
    AbstractCustomSource{M} <: AbstractOpticalSource{M}

Abstract type for stateful optical sources with custom generation logic.

Custom sources maintain internal state and support gradient computation for 
trainable source parameters like beam profiles, power levels, etc.

# Type Parameter
- `M::Trainability`: Determines gradient and buffer management

# Required Methods
All subtypes must implement:
- `propagate(source)`: Generate the optical field
- `get_data(source)`: Access to trainable parameters
- `trainable(source)`: Return trainable parameters (if `M <: Trainable`)

# Optional Methods (for optimization)
- `get_preallocated_gradient(source)`: Pre-allocated gradients (if buffered)
- `backpropagate_with_gradient(∂v, ∂p, source)`: Gradient backpropagation

See also: [`AbstractPureSource`](@ref), [`ScalarSource`](@ref)
"""
abstract type AbstractCustomSource{M} <: AbstractOpticalSource{M} end

function alloc_gradient(p::AbstractCustomSource{Trainable{Unbuffered}})
    map(similar, trainable(p))
end

function get_preallocated_gradient(p::AbstractCustomSource{Trainable{Buffered}})
    error("Not implemented")
end

function propagate_and_save(p::AbstractCustomSource{Trainable{Buffered}})
    error("Not implemented")
end

function backpropagate_with_gradient(∂v, ∂p::NamedTuple,
                                     p::AbstractCustomSource{<:Trainable})
    error("Not implemented")
end

function conj_direction(mask, ::Type{Forward})
    mask
end

function conj_direction(mask, ::Type{Backward})
    conj(mask)
end

include("sources/scalar_source.jl")
include("sources/helmholtz_source.jl")
export ScalarSource, HelmholtzSource, get_source

include("modulators/phasemask.jl")
export Phase

include("modulators/mask.jl")
export Mask

include("modulators/tea_doe.jl")
export TeaDOE, TeaReflector

include("modulators/helmholtz_index_slice.jl")
export HelmholtzIndexSlice

include("system/optical_sequence.jl")
export AbstractSequence, OpticalSequence, get_sequence

include("fourier/fourier_operator.jl")
export FourierOperator

include("fourier/fourier_wrapper.jl")
export FourierWrapper, FourierPhase, FourierMask

include("utilities/pad_crop_operators.jl")
export pad, crop, PadCropOperator

include("utilities/tilt_anchor.jl")
export TiltAnchor

include("freespace_propagators/freespace.jl")
export ASProp, ASPropZ, ShiftProp
export RSProp, CollinsProp, FourierLens, ParaxialProp
export as_rotation!, as_rotation, field_rotation_matrix
export HelmholtzProp, HelmholtzBoundedProp

include("bulk_propagators/bulk_propagators.jl")
export BPM, AS_BPM, TiltedAS_BPM, Shift_BPM
export FS_WPM, smoothstep_partition
export HelmholtzBPM, BidirectionalBPM

include("utilities/basis_projection_wrapper.jl")
export BasisProjectionWrapper, make_spatial_basis, make_fourier_basis

include("utilities/fourier_smoothing_wrapper.jl")
export FourierSmoothingWrapper

include("utilities/density_wrapper.jl")
export DensityWrapper, set_sharpness!, set_binarize!

include("active_media/active_media.jl")
export GainSheet

include("system/field_probe.jl")
export FieldProbe

include("system/merge_rules.jl")

include("system/optical_system.jl")
export OpticalSystem, get_source, get_components

end
