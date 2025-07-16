module Types

export Direction, Forward, Backward
export Trainability, Trainable, Static
export AbstractOpticalComponent

export adapt_2D

abstract type Direction end
struct Forward <: Direction end
struct Backward <: Direction end

abstract type Trainability end
struct Trainable <: Trainability end
struct Static <: Trainability end

is_trainable(::Type{Trainable}) = true
is_trainable(::Type{Static}) = false

abstract type AbstractOpticalComponent{T} end

function adapt_2D(A::Type{<:AbstractArray{T, N}}, f = identity) where {T, N}
    @assert N >= 2
    @assert isconcretetype(A)
    A.name.wrapper{f(A.parameters[1]), 2, A.parameters[3:end]...}
end

end
