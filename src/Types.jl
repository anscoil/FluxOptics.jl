module Types

export Direction, Forward, Backward
export Trainability, Trainable, Static
export AbstractOpticalComponent

export adapt_dim

abstract type Direction end
struct Forward <: Direction end
struct Backward <: Direction end

abstract type Trainability end
struct Trainable <: Trainability end
struct Static <: Trainability end

is_trainable(::Type{Trainable}) = true
is_trainable(::Type{Static}) = false

abstract type AbstractOpticalComponent{M} end

function adapt_dim(A::Type{<:AbstractArray{T}}, n::Integer, f = identity) where {T}
    @assert isconcretetype(A)
    A.name.wrapper{f(A.parameters[1]), n, A.parameters[3:end]...}
end

end
