abstract type AbstractSequence{M} <: AbstractPureComponent{M} end

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

function propagate!(u::ScalarField, p::AbstractSequence, direction::Type{<:Direction})
    for c in get_sequence(p)
        u = propagate!(u, c, direction)
    end
    u
end

function propagate(u::ScalarField, p::AbstractSequence, direction::Type{<:Direction})
    propagate!(copy(u), p, direction)
end

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
