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
    ) where {U <: AbstractArray{<:Complex}, S <: ScalarField{U}}
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

function propagate(p::ScalarSource)
    copyto!(p.uf.data, p.u0.data)
    p.uf
end

propagate_and_save(p::ScalarSource) = propagate(p)

function backpropagate_with_gradient(∂v, ∂p::NamedTuple, p::ScalarSource{<:Trainable})
    copyto!(∂p.u0.data, ∂v.data)
    ∂p
end

get_data(p::ScalarSource) = p.u0.data

function Base.fill!(p::ScalarSource, u0::ScalarField)
    copyto!(p.u0.data, u0.data)
end

function get_source(p::ScalarSource)
    p.u0
end
