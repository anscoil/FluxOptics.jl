"""
    OpticalSystem(source, components...; merge_components=false, inplace=false)
    source |> component1 |> component2 |> ... |> (; merge_components=true)

Create a complete optical system with source and components.

Represents a full optical system from source to output, combining a source
component with a sequence of transformations. Systems are callable and can
be optimized end-to-end.

# Arguments
- `source`: Source component (generates initial field)
- `components...`: Sequence of optical components
- `merge_components::Bool`: Merge adjacent compatible components for efficiency (default: false)
- `inplace::Bool`: Modify fields in-place during propagation (default: false)

# Component Merging

When `merge_components=true`, adjacent components of the same type are automatically
merged where possible (e.g., two static Phase masks → single Phase, two Fourier propagators → combined).
This can significantly improve performance by reducing the number of operations.

**Note:** After merging, `get_components(system)` returns the merged components,
which may differ from the original input components.

# Examples
```julia
# Basic system
source = ScalarSource(u0; trainable=true)
phase = Phase(u, (x, y) -> 0.0; trainable=true)
lens = FourierLens(u, (2.0, 2.0), 1000.0)
prop = ASProp(u, 500.0)

system = source |> phase |> lens |> prop

# With component merging (more efficient)
phase1 = Phase(u, (x, y) -> x^2)
phase2 = Phase(u, (x, y) -> y^2)
system_merged = source |> phase1 |> phase2 |> (; merge_components=true)
# phase1 and phase2 combined into single Phase

components = get_components(system_merged)  # Returns merged version

# Execute system
result = system()
output = result.out

# With probes
probe = FieldProbe()
system = source |> phase |> probe |> lens |> prop
result = system()
field_after_phase = result.probes[probe]

# In-place mode (memory efficient)
system_inplace = system |> (; inplace=true)
```

See also: [`OpticalSequence`](@ref), [`get_source`](@ref), [`get_components`](@ref)
"""
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

"""
    get_source(system::OpticalSystem)

Extract the source component from an optical system.

Returns the source component (e.g., `ScalarSource`) used to generate
the initial field in the system.

# Examples
```julia
system = source |> phase |> lens |> propagator

src = get_source(system)
```

See also: [`OpticalSystem`](@ref), [`get_components`](@ref)
"""
function get_source(p::OpticalSystem)
    p.source
end

"""
    get_components(system::OpticalSystem)

Extract the component sequence from an optical system.

Returns the tuple of components (excluding the source) that make up the system.
If the system was created with `merge_components=true`, returns the merged
components, which may differ from the original input.

# Examples
```julia
system = source |> phase |> lens |> propagator

components = get_components(system)  # (phase, lens, propagator)

# With merging
phase1 = Phase(u, (x, y) -> x^2)
phase2 = Phase(u, (x, y) -> y^2)
system_merged = source |> phase1 |> phase2 |> (; merge_components=true)

components_merged = get_components(system_merged)  # Single merged Phase
```

See also: [`get_source`](@ref), [`OpticalSystem`](@ref)
"""
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
