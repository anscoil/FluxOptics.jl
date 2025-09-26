struct OpticalSequence{M, C} <: AbstractPureComponent{M}
    optical_components::C

    function OpticalSequence(optical_components::C) where {
            N, C <: NTuple{N, AbstractPipeComponent}}
        new{Trainable, C}(optical_components)
    end

    function OpticalSequence(optical_components::Vararg{AbstractPipeComponent})
        M = any(istrainable, optical_components) ? Trainable : Static
        C = typeof(optical_components)
        new{M, C}(optical_components)
    end
end

Functors.@functor OpticalSequence (optical_components,)

Base.length(p::OpticalSequence) = length(p.optical_components)

function get_data(p::OpticalSequence)
    data = filter(x -> x != (), Functors.fleaves(map(c -> get_data(c), p.optical_components)))
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

function Base.merge(p1::AbstractPipeComponent, p2::AbstractPipeComponent)
    OpticalSequence(p1, p2)
end

function Base.merge(p1::OpticalSequence, p2::AbstractPipeComponent)
    OpticalSequence(p1.optical_components..., p2)
end

function Base.merge(p1::AbstractPipeComponent, p2::OpticalSequence)
    OpticalSequence(p1, p2.optical_components...)
end

function Base.merge(p::OpticalSequence)
    n = length(p.optical_components)
    if n < 2
        p
    elseif n == 2
        merge(p.optical_components...)
    else
        p_merged = merge(OpticalSequence(), p)
        while p_merged != p
            p = p_merged
            p_merged = merge(OpticalSequence(), p)
        end
        p_merged
    end
end

function Base.merge(p1::OpticalSequence, p2::OpticalSequence)
    n1 = length(p1.optical_components)
    n2 = length(p2.optical_components)
    if n2 == 0
        p1
    else
        if n1 > 0
            p_merged = merge(last(p1.optical_components), first(p2.optical_components))
        else
            p_merged = OpticalSequence(first(p2.optical_components))
        end
        head = OpticalSequence(p1.optical_components[1:(n1 - 1)]..., p_merged.optical_components...)
        tail = OpticalSequence(p2.optical_components[2:end]...)
        merge(head, tail)
    end
end
