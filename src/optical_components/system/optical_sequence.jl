"""
    AbstractSequence{M} <: AbstractPipeComponent{M}

Abstract type for sequences of optical components.

Sequences combine multiple components into a single composite component that
can be used like any other pipe component. The type parameter `M` reflects
the combined trainability of all components in the sequence.

# Subtypes
- [`OpticalSequence`](@ref): Concrete sequence implementation

# Required Interface
All subtypes must implement:
- [`get_sequence(seq::AbstractSequence)`](@ref)

See also: [`OpticalSequence`](@ref), [`get_sequence`](@ref)
"""
abstract type AbstractSequence{M} <: AbstractPureComponent{M} end

"""
    get_sequence(seq::AbstractSequence)

Extract the tuple of components from a sequence.

Returns the ordered tuple of components that make up the sequence.

# Examples
```julia
seq = OpticalSequence(phase1, lens, phase2)
components = get_sequence(seq)  # (phase1, lens, phase2)
```

See also: [`AbstractSequence`](@ref), [`OpticalSequence`](@ref)
"""
function get_sequence(p::AbstractSequence)
    error("Not Implemented")
end

Base.length(p::AbstractSequence) = sum(map(length, get_sequence(p)))

function get_data(p::AbstractSequence)
    data = filter(x -> !isempty(x),
                  Functors.fleaves(map(c -> get_data(c), get_sequence(p))))
    if length(data) == 1
        first(data)
    else
        Tuple(data)
    end
end

trainable(p::AbstractSequence{<:Trainable}) = (; optical_components = get_sequence(p))

function propagate!(u::ScalarField, p::AbstractSequence)
    _propagate_sequence!(u, get_sequence(p))
end

function backpropagate!(u::ScalarField, p::AbstractSequence)
    _backpropagate_sequence!(u, get_sequence(p))
end

@inline _propagate_sequence!(u, ::Tuple{}) = u

@inline _backpropagate_sequence!(u, ::Tuple{}) = u

@inline function _propagate_sequence!(u, components::Tuple)
    u = propagate!(u, first(components))
    _propagate_sequence!(u, Base.tail(components))
end

@inline function _backpropagate_sequence!(u, components::Tuple)
    u = backpropagate!(u, last(components))
    _backpropagate_sequence!(u, Base.front(components))
end

propagate(u::ScalarField, p::AbstractSequence) = propagate!(copy(u), p)

backpropagate(u::ScalarField, p::AbstractSequence) = backpropagate!(copy(u), p)

"""
    OpticalSequence(components...)

Create a sequence of optical components without a source.

Stores multiple pipe components as a single composite component. Unlike `OpticalSystem`,
this does not include a source and cannot use the pipe operator syntax.

**Note:** The pipe operator (`|>`) only works for `OpticalSystem`, not `OpticalSequence`.
To create sequences, pass components directly to the constructor.

# Arguments
- `components...`: Sequence of `AbstractPipeComponent` instances

# Examples
```julia
# Create sequence explicitly
u = ScalarField(ones(ComplexF64, 256, 256), (2.0, 2.0), 1.064)

phase = Phase(u, (x, y) -> x^2)
lens = FourierLens(u, (2.0, 2.0), 1000.0)
prop = ASProp(u, 500.0)

sequence = OpticalSequence(phase, lens, prop)

# Apply to field
result = propagate(u, sequence)

# Extract components
components = get_sequence(sequence)
```

See also: [`OpticalSystem`](@ref), [`AbstractSequence`](@ref), [`get_sequence`](@ref)
"""
struct OpticalSequence{M, C} <: AbstractSequence{M}
    optical_components::C

    function OpticalSequence(optical_components::C) where {N,
                                                           C <:
                                                           NTuple{N, AbstractPipeComponent}}
        new{Trainable, C}(optical_components)
    end

    function OpticalSequence(optical_components::Vararg{AbstractPipeComponent})
        M = any(istrainable, optical_components) ? Trainable : Static
        C = typeof(optical_components)
        new{M, C}(optical_components)
    end
end

Functors.@functor OpticalSequence (optical_components,)

get_sequence(p::OpticalSequence) = p.optical_components
