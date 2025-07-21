module Types

export Direction, Forward, Backward
export Trainability, Trainable, Static
export AbstractOpticalComponent

export is_trainable, has_prealloc
export adapt_dim

abstract type Direction end
struct Forward <: Direction end
struct Backward <: Direction end

function Base.reverse(::Type{Forward})
    Backward
end

function Base.reverse(::Type{Backward})
    Forward
end

abstract type Trainability end
struct Static <: Trainability end
struct Trainable{A <: Union{Nothing, <:NamedTuple}} <: Trainability end

is_trainable(::Type{<:Trainable}) = true
is_trainable(::Type{Static}) = false

has_prealloc(::Type{Trainable{Nothing}}) = false
has_prealloc(::Type{Trainable{<:NamedTuple}}) = true

abstract type AbstractOpticalComponent{M <: Trainability} end

function adapt_dim(A::Type{<:AbstractArray{T}}, n::Integer, f = identity) where {T}
    @assert isconcretetype(A)
    A.name.wrapper{f(A.parameters[1]), n, A.parameters[3:end]...}
end

end
