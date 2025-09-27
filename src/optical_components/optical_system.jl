struct OpticalSystem
    source::Union{Nothing, AbstractOpticalSource}
    components::OpticalSequence
    direction::Type{<:Direction}
    inplace::Bool
    merge_components::Bool

    function OpticalSystem(source, components, direction, inplace, merge_components)
        new(source, components, direction, inplace, merge_components)
    end

    function OpticalSystem(source::Union{Nothing, AbstractOpticalSource},
                           components::Vararg{AbstractPipeComponent};
                           inplace::Bool = false,
                           direction::Type{<:Direction} = Forward,
                           merge_components::Bool = false)
        components = OpticalSequence(components...)
        if merge_components
            components = merge(components)
        end
        new(source, components, direction, inplace, merge_components)
    end

    function OpticalSystem(components::Vararg{AbstractPipeComponent};
                           inplace::Bool = false,
                           direction::Type{<:Direction} = Forward,
                           merge_components::Bool = false)
        OpticalSystem(nothing, components...; inplace, direction, merge_components)
    end
end

Functors.@functor OpticalSystem (source, components)

function Base.length(p::OpticalSystem)
    length(p.components)
end

function get_source(p::OpticalSystem)
    p.source
end

function get_components(p::OpticalSystem)
    p.components.optical_components
end

function Base.:|>(source::AbstractOpticalSource, component::AbstractPipeComponent)
    OpticalSystem(source, component)
end

function Base.:|>(source::AbstractOpticalSource, system::OpticalSystem)
    OpticalSystem(source, system.components.optical_components...)
end

function Base.:|>(c1::AbstractPipeComponent, c2::AbstractPipeComponent)
    OpticalSystem(c1, c2)
end

function Base.:|>(component::AbstractPipeComponent, system::OpticalSystem)
    OpticalSystem(component, system.components.optical_components...;
                  direction = system.direction, inplace = system.inplace,
                  system.merge_components)
end

function Base.:|>(system::OpticalSystem, component::AbstractPipeComponent)
    OpticalSystem(system.source, system.components.optical_components..., component;
                  direction = system.direction, inplace = system.inplace,
                  system.merge_components)
end

function Base.:|>(s1::OpticalSystem, s2::OpticalSystem)
    @assert s1.direction == s2.direction
    @assert s1.inplace == s2.inplace
    @assert isnothing(s2.source)
    merge_components = s1.merge_components && s2.merge_components
    OpticalSystem(s1.source, s1.components.optical_components...,
                  s2.components.optical_components...;
                  direction = s1.direction, inplace = s1.inplace, merge_components)
end

function Base.:|>(system::OpticalSystem, kwargs::NamedTuple)
    OpticalSystem(system.source, system.components.optical_components...; kwargs...)
end

function Base.:|>(component::AbstractOpticalComponent, kwargs::NamedTuple)
    OpticalSystem(component; kwargs...)
end

function compute_split_output(p::AbstractPipeComponent, u, inplace, direction)
    v = inplace ? propagate!(u, p, direction) : propagate(u, p, direction)
    (v, nothing)
end

function compute_split_output(p::FieldProbe, u, inplace, direction)
    propagate(u, p, direction)
end

function iter_components(components, x, d, inplace, direction)
    for component in components
        x, x_probe = compute_split_output(component, x, inplace, direction)
        if !isnothing(x_probe)
            d[component] = x_probe
        end
    end
    (; out = x, probes = d)
end

function (system::OpticalSystem)(x::Union{Nothing, ScalarField} = nothing)
    @assert (isnothing(x) ⊻ isnothing(system.source))
    x = isnothing(x) ? propagate(system.source) : x
    components = system.components.optical_components
    d = IdDict{AbstractOpticalComponent, typeof(x)}()
    iter_components(components, x, d, system.inplace, system.direction)
end
