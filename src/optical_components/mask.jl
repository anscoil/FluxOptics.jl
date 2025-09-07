struct Mask{M, A} <: AbstractPureComponent{M}
    m::A

    function Mask(m::A) where {A}
        new{Trainable, A}(m)
    end

    function Mask(u::ScalarField{U, Nd},
            ds::NTuple{Nd, Real},
            f::Function = (_...) -> 1;
            trainable::Bool = false
    ) where {Nd, U <: AbstractArray{<:Complex}}
        ns = size(u.data)[1:Nd]
        xs = spatial_vectors(ns, ds)
        m = Nd == 2 ? f.(xs[1], xs[2]') : f.(xs[1])
        A = adapt_dim(U, Nd, eltype(m) <: Real ? real : complex)
        M = trainable ? Trainable : Static
        new{M, A}(A(m))
    end

    function Mask(u::ScalarField,
            f::Function = (_...) -> 1;
            trainable::Bool = false)
        Mask(u, u.ds, f; trainable)
    end
end

Functors.@functor Mask (m,)

get_data(p::Mask) = p.m

trainable(p::Mask{<:Trainable}) = (; m = p.m)

function propagate(u::AbstractArray, p::Mask, direction::Type{<:Direction})
    u .* conj_direction(p.m, direction)
end

function propagate(u::ScalarField, p::Mask, direction::Type{<:Direction})
    data = u.data .* conj_direction(p.m, direction)
    ScalarField(data, u.ds, u.lambdas, u.lambdas_collection)
end
