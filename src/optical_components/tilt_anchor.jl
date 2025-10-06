"""
    TiltAnchor(u::ScalarField; trainable=false)

Anchor point for tilt reference in optical systems.

Maintains a reference for beam tilt through a sequence of components. Useful
in off-axis systems where tilt changes need to be tracked or reset.

# Arguments
- `u::ScalarField`: Field template
- `trainable::Bool`: Typically false (default: false)

# Examples
```julia
# Anchor tilt at specific points
anchor1 = TiltAnchor(u)
anchor2 = TiltAnchor(u)

# System with tilt anchors
system = source |> anchor1 |> components... |> anchor2

# Maintains tilt reference through propagation
```

See also: [`offset_tilts!`](@ref), [`set_field_tilts`](@ref)
"""
struct TiltAnchor{M, A} <: AbstractCustomComponent{M}
    tilts::A
    tilts_saved::Union{Nothing, A}

    function TiltAnchor(tilts::A, tilts_saved::Union{Nothing, A}) where {A}
        M = isnothing(tilts_saved) ? Trainable{Unbuffered} : Trainable{Buffered}
        new{M, A}(tilts, tilts_saved)
    end

    function TiltAnchor(u::ScalarField; trainable::Bool = false, buffered::Bool = false)
        M = trainability(trainable, buffered)
        tilts = map(copy, u.tilts.collection)
        tilts_saved = (trainable && buffered) ? map(similar, tilts) : nothing
        A = typeof(tilts)
        new{M, A}(tilts, tilts_saved)
    end
end

Functors.@functor TiltAnchor ()

trainable(p::TiltAnchor{<:Trainable}) = (;)

get_preallocated_gradient(p::TiltAnchor{Trainable{Buffered}}) = (;)

function alloc_saved_buffer(u::ScalarField, p::TiltAnchor{Trainable{Unbuffered}})
    map(similar, u.tilts.collection)
end

get_saved_buffer(p::TiltAnchor{Trainable{Buffered}}) = p.tilts_saved

function propagate!(u, p::TiltAnchor, ::Type{<:Direction})
    offset_tilts!(u, p.tilts)
end

function propagate_and_save!(u, p::TiltAnchor{Trainable{Buffered}},
                             direction::Type{<:Direction})
    foreach(((y, x),) -> copyto!(y, x), zip(p.tilts_saved, u.tilts.collection))
    u
end

function propagate_and_save!(u, tilts_saved, p::TiltAnchor{Trainable{Unbuffered}},
                             ::Type{<:Direction})
    foreach(((y, x),) -> copyto!(y, x), zip(tilts_saved, u.tilts.collection))
    u
end

function backpropagate!(∂v, p::TiltAnchor, ::Type{<:Direction})
    offset_tilts!(∂v, p.tilts)
end

function backpropagate_with_gradient!(∂v, tilts_saved, ∂p::NamedTuple,
                                      p::TiltAnchor{<:Trainable}, ::Type{<:Direction})
    @assert ∂p == (;)
    offset_tilts!(∂v, tilts_saved), ∂p
end
