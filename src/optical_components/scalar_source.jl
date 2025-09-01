struct ScalarSource{M, S} <: AbstractCustomSource{M}
    u0::S
    uf::S
    ∂p::Union{Nothing, @NamedTuple{u0::S}}

    function ScalarSource(u0::S, uf::S, ∂p) where {S}
        new{Static, S}(u0, uf, ∂p)
    end

    function ScalarSource(u::S;
            trainable::Bool = false,
            buffered::Bool = false
    ) where {U <: AbstractArray{<:Complex}, S <: Union{U, ScalarField{U}}}
        u0 = copy(u)
        uf = similar(u)
        M = trainability(trainable, buffered)
        ∂p = (trainable && buffered) ? (; u0 = similar(u0)) : nothing
        new{M, S}(u0, uf, ∂p)
    end
end

Functors.@functor ScalarSource (u0,)

Base.collect(p::ScalarSource) = collect(p.u0)
Base.size(p::ScalarSource) = size(p.u0)

trainable(p::ScalarSource{<:Trainable}) = (; u0 = p.u0)

get_preallocated_gradient(p::ScalarSource{Trainable{Buffered}}) = p.∂p

function propagate(p::ScalarSource, direction::Type{<:Direction})
    copyto!(get_data(p.uf), get_data(p.u0))
    p.uf
end

function propagate_and_save(p::ScalarSource, direction::Type{<:Direction})
    propagate(p, direction)
end

function backpropagate_with_gradient(∂v, ∂p::NamedTuple, p::ScalarSource{<:Trainable},
        direction::Type{<:Direction})
    copyto!(get_data(∂p.u0), get_data(∂v))
    ∂p
end

function Base.fill!(p::ScalarSource,
        u0::Union{U, ScalarField{U}}
) where {U <: AbstractArray{<: Complex}}
    copyto!(get_data(p.u0), get_data(u0))
end

function get_source(p::ScalarSource)
    p.u0
end
