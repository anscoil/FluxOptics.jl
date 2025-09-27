struct OpticalSequence{M, C} <: AbstractPureComponent{M}
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

Base.length(p::OpticalSequence) = sum(map(length, p.optical_components))

function get_data(p::OpticalSequence)
    data = filter(x -> isa(x, AbstractArray),
                  Functors.fleaves(map(c -> get_data(c), p.optical_components)))
    if length(data) == 1
        first(data)
    else
        Tuple(data)
    end
end

trainable(p::OpticalSequence{<:Trainable}) = (; optical_components = p.optical_components)

function propagate!(u::ScalarField, p::OpticalSequence, direction::Type{<:Direction})
    for c in p.optical_components
        u = propagate!(u, c, direction)
    end
    u
end

function propagate(u::ScalarField, p::OpticalSequence, direction::Type{<:Direction})
    propagate!(copy(u), p, direction)
end
