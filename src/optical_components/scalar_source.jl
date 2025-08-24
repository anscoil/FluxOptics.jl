struct ScalarSource{M, U} <: AbstractOpticalSource{M}
    u0::U
    uf::U
    ∂p::Union{Nothing, @NamedTuple{u0::U}}

    function ScalarSource(u::U;
            trainable::Bool = false,
            prealloc_gradient::Bool = false
    ) where {U <: Union{ScalarField, AbstractArray{<:Complex}}}
        u0 = copy(u)
        uf = similar(u)

        M = trainable ?
            (prealloc_gradient ?
             Trainable{@NamedTuple{u0::U}} :
             Trainable{Nothing}) :
            Static

        ∂p = (trainable && prealloc_gradient) ? (; u0 = similar(u0)) : nothing

        new{M, U}(u0, uf, ∂p)
    end
end

trainable(p::ScalarSource{<:Trainable}) = (; u0 = get_data(p.u0))

get_preallocated_gradient(p::ScalarSource{<:Trainable{<:NamedTuple}}) = p.∂p

function propagate(p::ScalarSource, direction::Type{<:Direction})
    copyto!(p.uf, p.u0)
    p.uf
end

function propagate_and_save(p::ScalarSource, direction::Type{<:Direction})
    propagate(p, direction)
end

function backpropagate_with_gradient!(∂v, ∂p::NamedTuple, p::ScalarSource{<:Trainable},
        direction::Type{<:Direction})
    copyto!(∂p.u0, ∂v)
    ∂p
end
