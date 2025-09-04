module OpticalComponents

using Functors
using AbstractFFTs
using EllipsisNotation
using LRUCache
using ..FluxOptics: get_data, intensity
using ..GridUtils
using ..Fields
using ..FFTutils

export Direction, Forward, Backward
export AbstractOpticalComponent, AbstractOpticalSource
export AbstractCustomComponent, AbstractCustomSource
export AbstractPureComponent, AbstractPureSource
export propagate!, propagate
export alloc_saved_buffer, get_saved_buffer

abstract type Direction end
struct Forward <: Direction end
struct Backward <: Direction end

Base.reverse(::Type{Forward}) = Backward
Base.reverse(::Type{Backward}) = Forward
Base.sign(::Type{Forward}) = 1
Base.sign(::Type{Backward}) = -1

abstract type Buffering end
struct Buffered <: Buffering end
struct Unbuffered <: Buffering end

abstract type Trainability end
struct Static <: Trainability end
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
            @warn "Invalid combination: `bufferd=true` only makes sense when \
            `trainable=true`.\nIgnoring buffering."
        end
        Static
    end
end

abstract type AbstractOpticalComponent{M <: Trainability} end

function Base.fill!(p::AbstractOpticalComponent, v)
    error("Not implemented")
end

trainable(p::AbstractOpticalComponent{Static}) = NamedTuple{}()

function trainable(p::AbstractOpticalComponent{<:Trainable})
    error("Not implemented")
end

abstract type AbstractCustomComponent{M} <: AbstractOpticalComponent{M} end

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

function backpropagate!(u, p::AbstractCustomComponent, direction::Type{<:Direction})
    error("Not implemented")
end

function backpropagate_with_gradient!(
        ∂v, u_saved, ∂p::NamedTuple, p::AbstractCustomComponent{<:Trainable},
        direction::Type{<:Direction})
    error("Not implemented")
end

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

function backpropagate(u, p::AbstractCustomComponent, direction::Type{<:Direction})
    backpropagate!(copy(u), p, direction)
end

function backpropagate_with_gradient(
        ∂v, u_saved, ∂p::NamedTuple, p::AbstractCustomComponent{<:Trainable},
        direction::Type{<:Direction})
    backpropagate_with_gradient!(copy(∂v), u_saved, ∂p, p, direction)
end

abstract type AbstractPureComponent{M} <: AbstractOpticalComponent{M} end

function propagate(u, p::AbstractPureComponent, direction::Type{<:Direction})
    error("Not implemented")
end

function propagate!(u, p::AbstractPureComponent, direction::Type{<:Direction})
    propagate(u, p, direction)
end

abstract type AbstractOpticalSource{M} <: AbstractOpticalComponent{M} end

function propagate(p::AbstractOpticalSource)
    error("Not implemented")
end

abstract type AbstractPureSource{M} <: AbstractOpticalSource{M} end

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

function backpropagate_with_gradient(
        ∂v, ∂p::NamedTuple, p::AbstractCustomSource{<:Trainable})
    error("Not implemented")
end

function conj_direction(mask, ::Type{Forward})
    mask
end

function conj_direction(mask, ::Type{Backward})
    conj(mask)
end

include("freespace_propagators/freespace.jl")
export ASProp, ASPropZ, TiltedASProp, RSProp, CollinsProp, FourierLens, ParaxialProp

include("scalar_source.jl")
export ScalarSource, get_source

include("phasemask.jl")
export Phase

include("mask.jl")
export Mask

include("tea_doe.jl")
export TeaDOE, TeaReflector

include("bulk_propagators/bulk_propagators.jl")
export BPM, AS_BPM, TiltedAS_BPM

include("field_probe.jl")
export FieldProbe

include("active_media/active_media.jl")
export GainSheet

end
