module OpticalComponents

using Functors
using LinearAlgebra
using AbstractFFTs
using FINUFFT
using EllipsisNotation
using LRUCache
using ..FluxOptics: isbroadcastable, bzip
using ..GridUtils
using ..Fields
using ..FFTutils

import Zygote: pullback

export Direction, Forward, Backward
export Trainability, Trainable, Static, Buffering, Buffered, Unbuffered
export AbstractOpticalComponent, AbstractPipeComponent, AbstractOpticalSource
export AbstractCustomComponent, AbstractCustomSource
export AbstractPureComponent, AbstractPureSource
export propagate!, propagate
export backpropagate!, backpropagate
export alloc_saved_buffer, get_saved_buffer
export get_data
export istrainable, isbuffered

abstract type Direction end

"""
    Forward

Direction marker type for forward optical propagation.

Used as a type parameter in propagation functions to indicate the direction
of light propagation. Forward propagation follows the positive z-axis direction.

See also: [`Backward`](@ref), [`propagate!`](@ref)
"""
struct Forward <: Direction end

"""
    Backward

Direction marker type for backward optical propagation.

Used as a type parameter in propagation functions to indicate the direction
of light propagation along the negative z-axis direction.

See also: [`Forward`](@ref), [`propagate!`](@ref)
"""
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

See also: [`Trainable`](@ref), [`trainability`](@ref)
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

See also: [`Unbuffered`](@ref), [`trainability`](@ref)
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

See also: [`Buffered`](@ref), [`trainability`](@ref)
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

See also: [`Buffering`](@ref), [`trainability`](@ref)
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

See also: [`Trainable`](@ref), [`trainability`](@ref)
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

See also: [`Static`](@ref), [`Buffering`](@ref), [`trainable`](@ref)
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

istrainable(p::AbstractOpticalComponent{Static}) = false
istrainable(p::AbstractOpticalComponent{<:Trainable}) = true
isbuffered(p::AbstractOpticalComponent) = false
isbuffered(p::AbstractOpticalComponent{Trainable{Buffered}}) = false

function get_data(p::AbstractOpticalComponent)
    error("Not implemented")
end

function Base.collect(p::AbstractOpticalComponent)
    data = get_data(p)
    if isa(data, Tuple)
        map(collect, data)
    else
        collect(data)
    end
end

Base.length(p::AbstractOpticalComponent) = 1

Base.size(p::AbstractOpticalComponent) = size(get_data(p))

function Base.fill!(p::AbstractOpticalComponent, v::Real)
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
    if isa(data, Tuple)
        foreach(data -> isa(data, AbstractArray) ? copyto!(data, v) : nothing, get_data(p))
    else
        copyto!(data, v)
    end
    data
end

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
- `propagate!(u, component, direction)` or `propagate(u, component, direction)`
- `get_data(component)`: Access to component parameters

See also: [`AbstractOpticalSource`](@ref), [`propagate!`](@ref), [`|>`](@ref)
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
- `M::Trainability`: Usually `Static`, but can support `Trainable` for Zygote-based components

# Required Methods
All subtypes must implement:
- `propagate(u, component, direction)`: Direct field transformation
- `get_data(component)`: Access to component parameters

# Characteristics
- **Functional interface**: Same input → same output, regardless of internal complexity
- **Zygote compatible**: Automatic differentiation works without custom rules
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

function propagate(u, p::AbstractPureComponent, direction::Type{<:Direction})
    error("Not implemented")
end

function propagate!(u, p::AbstractPureComponent, direction::Type{<:Direction})
    propagate(u, p, direction)
end

"""
    backpropagate!(∂v::ScalarField, component::AbstractPipeComponent, direction::Type{<:Direction})

Backpropagate gradients through an optical component in-place.

Computes the adjoint (reverse-mode) propagation of gradients through the component,
modifying `∂v` in-place to contain the gradient with respect to the input field.
This is useful for debugging gradient flow, educational purposes, or manual gradient
computation outside of automatic differentiation frameworks.

For `AbstractPureComponent`s, this is implemented automatically using Zygote's pullback.
For `AbstractCustomComponent`s, this must be implemented manually for each component type.

# Arguments
- `∂v::ScalarField`: Gradient with respect to output field (modified in-place to become ∂u)
- `component::AbstractPipeComponent`: Optical component to backpropagate through
- `direction::Type{<:Direction}`: Original propagation direction (`Forward` or `Backward`)

# Returns
The modified `∂v`, now containing the gradient with respect to the input field.

# Notes
- The `direction` parameter specifies the **original forward direction**, not the backprop direction
- For `AbstractPureComponent`, uses automatic differentiation via Zygote
- For `AbstractCustomComponent`, requires manual implementation of adjoint propagation
- Only computes gradient with respect to the input field, not the component parameters

# Examples
```jldoctest
julia> w0 = 10.0;

julia> xv, yv = spatial_vectors(64, 64, 2.0, 2.0);

julia> u = ScalarField(Gaussian(w0)(xv, yv), (2.0, 2.0), 1.064);

julia> phase_mask = Phase(u, (x, y) -> 0.01 * (x^2 + y^2));

julia> propagator1 = ASProp(u, 200.0);

julia> propagator2 = ASProp(u, 300.0);

julia> sequence = OpticalSequence(propagator1, phase_mask, propagator2);

julia> v = propagate(u, sequence, Forward);

julia> ∂v = copy(v);

julia> ∂u = backpropagate!(∂v, sequence, Forward);

julia> # Coupling efficiency ≈ 1 demonstrates unitary optical propagation

julia> all(x -> isapprox(x, 1), coupling_efficiency(u, ∂u))
true

julia> # Step-by-step backprop for debugging

julia> v2 = propagate(u, sequence, Forward);

julia> ∂v2 = copy(v2);

julia> # Second propagator appearing first in reverse mode

julia> ∂after_prop2 = backpropagate!(∂v2, propagator2, Forward);

julia> ∂after_phase = backpropagate!(∂after_prop2, phase_mask, Forward);

julia> ∂u_step = backpropagate!(∂after_phase, propagator1, Forward);

julia> all(x -> isapprox(x, 1), coupling_efficiency(u, ∂u))
true
```

See also: [`backpropagate`](@ref), [`propagate!`](@ref), [`Forward`](@ref), [`Backward`](@ref)
"""
function backpropagate!(∂v::ScalarField,
                        p::AbstractPureComponent,
                        direction::Type{<:Direction})
    _, back = pullback(u -> propagate(u, p, direction), ∂v)
    ∂u, = back(∂v)
    ∂u
end

"""
    backpropagate(∂v::ScalarField, component::AbstractPipeComponent, direction::Type{<:Direction})

Backpropagate gradients through an optical component (non-mutating).

This is the non-mutating version of [`backpropagate!`](@ref). Creates a copy of the 
gradient field before computing the adjoint propagation.

See also: [`backpropagate!`](@ref), [`propagate`](@ref)
"""
function backpropagate(∂v::ScalarField,
                       p::AbstractPipeComponent,
                       direction::Type{<:Direction})
    backpropagate!(copy(∂v), p, direction)
end

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
- `propagate!(u, component, direction)`: In-place field transformation
- `get_data(component)`: Access to trainable parameters  
- `trainable(component)`: Return trainable parameters (if `M <: Trainable`)

# Optional Methods (for optimization)
- `backpropagate!(u, component, direction)`: Reverse propagation
- `get_preallocated_gradient(component)`: Pre-allocated gradients (if buffered)
- `alloc_saved_buffer(u, component)`: Allocate forward-pass buffers

# Examples
```jldoctest
julia> u = ScalarField(ones(ComplexF64, 64, 64), (2.0, 2.0), 1.064);

julia> phase_mask = Phase(u, (x, y) -> 0.1*x^2; trainable=true);

julia> typeof(phase_mask) <: AbstractCustomComponent
true

julia> propagate!(u, phase_mask, Forward);
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

"""
    propagate!(u::ScalarField, component, direction::Type{<:Direction})

Propagate an optical field through a component in-place.

Modifies the input field `u` by applying the optical transformation defined by
`component` in the specified `direction`. This is the core function for optical
propagation in FluxOptics.

# Arguments
- `u::ScalarField`: Input optical field (modified in-place)
- `component`: Optical component to propagate through
- `direction::Type{<:Direction}`: `Forward` or `Backward` propagation

# Returns
The modified field `u`.

# Examples
```jldoctest
julia> u = ScalarField(ones(ComplexF64, 64, 64), (2.0, 2.0), 1.064);

julia> lens = FourierLens(u, (2.0, 2.0), 1000.0);

julia> propagate!(u, lens, Forward);
```

See also: [`propagate`](@ref), [`Forward`](@ref), [`Backward`](@ref)
"""
function propagate!(u, p::AbstractCustomComponent, direction::Type{<:Direction})
    error("Not implemented")
end

function propagate_and_save!(u, p::AbstractCustomComponent{Trainable{Buffered}},
                             direction::Type{<:Direction})
    error("Not implemented")
end

function propagate_and_save!(u, u_saved, p::AbstractCustomComponent{Trainable{Unbuffered}},
                             direction::Type{<:Direction})
    error("Not implemented")
end

function backpropagate!(∂v, p::AbstractCustomComponent, direction::Type{<:Direction})
    error("Not implemented")
end

function backpropagate_with_gradient!(∂v, u_saved, ∂p::NamedTuple,
                                      p::AbstractCustomComponent{<:Trainable},
                                      direction::Type{<:Direction})
    error("Not implemented")
end

"""
    propagate(u::ScalarField, component, direction::Type{<:Direction})
    propagate(source::AbstractOpticalSource)

Propagate an optical field through a component or generate a field from a source.

The first form creates a copy of the input field and applies the optical transformation 
defined by `component` in the specified `direction`. The second form generates a new
optical field from a source component.

# Arguments
- `u::ScalarField`: Input optical field (unchanged)
- `component`: Optical component to propagate through  
- `direction::Type{<:Direction}`: `Forward` or `Backward` propagation
- `source::AbstractOpticalSource`: Source component to generate field from

# Returns
New `ScalarField` with the transformation applied or generated.

# Examples
```jldoctest
julia> u = ScalarField(ones(ComplexF64, 64, 64), (2.0, 2.0), 1.064);

julia> phase_mask = Phase(u, (x, y) -> 0.1*(x^2 + y^2));

julia> u_prop = propagate(u, phase_mask, Forward);

julia> source = ScalarSource(u; trainable=true);

julia> u_generated = propagate(source);

julia> size(u_generated) == size(u)
true
```

See also: [`propagate!`](@ref), [`Forward`](@ref), [`Backward`](@ref)
"""
function propagate(u, p::AbstractCustomComponent, direction::Type{<:Direction})
    propagate!(copy(u), p, direction)
end

function propagate(u::AbstractArray, p::AbstractCustomComponent,
                   λ::Real, direction::Type{<:Direction})
    propagate!(copy(u), p, λ, direction)
end

function propagate_and_save(u, p::AbstractCustomComponent{Trainable{Buffered}},
                            direction::Type{<:Direction})
    propagate_and_save!(copy(u), p, direction; saved_buffer)
end

function propagate_and_save(u, u_saved, p::AbstractCustomComponent{Trainable{Unbuffered}},
                            direction::Type{<:Direction})
    propagate_and_save!(copy(u), u_saved, p, direction; saved_buffer)
end

function backpropagate_with_gradient(∂v, u_saved, ∂p::NamedTuple,
                                     p::AbstractCustomComponent{<:Trainable},
                                     direction::Type{<:Direction})
    backpropagate_with_gradient!(copy(∂v), u_saved, ∂p, p, direction)
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

function propagate_and_save(p::AbstractCustomSource{Trainable{Buffered}},
                            direction::Type{<:Direction})
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

function function_to_array(f::Function, ns::NTuple{Nd, Integer}, ds::NTuple{Nd, Real},
                           isfourier = false) where {Nd}
    if isfourier
        xs = [fftfreq(nx, 1/dx) for (nx, dx) in zip(ns, ds)]
    else
        xs = spatial_vectors(ns, ds)
    end
    Nd == 2 ? f.(xs[1], xs[2]') : f.(xs[1])
end

include("scalar_source.jl")
export ScalarSource, get_source

include("phasemask.jl")
export Phase

include("mask.jl")
export Mask

include("tea_doe.jl")
export TeaDOE, TeaReflector

include("optical_sequence.jl")
export AbstractSequence, OpticalSequence

include("fourier_operator.jl")
export FourierOperator

include("fourier_wrapper.jl")
export FourierWrapper, FourierPhase, FourierMask

include("pad_crop_operators.jl")
export pad, crop, PadCropOperator

include("tilt_anchor.jl")
export TiltAnchor

include("freespace_propagators/freespace.jl")
export ASProp, ASPropZ, ShiftProp
export RSProp, CollinsProp, FourierLens, ParaxialProp
export as_rotation!, as_rotation, field_rotation_matrix

include("bulk_propagators/bulk_propagators.jl")
export BPM, AS_BPM, TiltedAS_BPM, Shift_BPM

include("field_probe.jl")
export FieldProbe

include("basis_projection_wrapper.jl")
export BasisProjectionWrapper, set_basis_projection!, make_spatial_basis, make_fourier_basis

include("active_media/active_media.jl")
export GainSheet

include("merge_rules.jl")

include("optical_system.jl")
export OpticalSystem, get_source, get_components

end
