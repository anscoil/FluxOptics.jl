struct ScalarSource{M, U, S} <: AbstractCustomSource{M}
    u0::U
    uf::S
    ∂p::Union{Nothing, @NamedTuple{u0::U}}

    function ScalarSource(u0::U, uf::S, ∂p) where {U, S}
        new{Static, U, S}(u0, uf, ∂p)
    end

    function ScalarSource(u::Union{U, ScalarField{U}};
            trainable::Bool = false,
            prealloc_gradient::Bool = false
    ) where {U <: AbstractArray{<:Complex}}
        u0 = copy(get_data(u))
        uf = similar(u)

        M = trainable ?
            (prealloc_gradient ?
             Trainable{GradAllocated} :
             Trainable{GradNoAlloc}) :
            Static

        ∂p = (trainable && prealloc_gradient) ? (; u0 = similar(u0)) : nothing

        new{M, U, typeof(uf)}(u0, uf, ∂p)
    end
end

Functors.@functor ScalarSource (u0,)

Base.collect(p::ScalarSource) = collect(p.u0)
Base.size(p::ScalarSource) = size(p.u0)

trainable(p::ScalarSource{<:Trainable}) = (; u0 = p.u0)

get_preallocated_gradient(p::ScalarSource{Trainable{GradAllocated}}) = p.∂p

function propagate(p::ScalarSource, direction::Type{<:Direction})
    copyto!(get_data(p.uf), p.u0)
    p.uf
end

function propagate_and_save(p::ScalarSource, direction::Type{<:Direction})
    propagate(p, direction)
end

function backpropagate_with_gradient!(∂v, ∂p::NamedTuple, p::ScalarSource{<:Trainable},
        direction::Type{<:Direction})
    copyto!(∂p.u0, get_data(∂v))
    ∂p
end

function init_source!(s::ScalarSource,
        u0::Union{U, ScalarField{U}}
) where {U <: AbstractArray{<: Complex}}
    copyto!(s.u0, get_data(u0))
end
