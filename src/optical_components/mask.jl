struct Mask{M, A} <: AbstractPureComponent{M}
    m::A

    function Mask(m::A) where {A}
        new{Trainable, A}(m)
    end

    function Mask(u::U,
            ds::NTuple{Nd, Real},
            f::Function;
            trainable::Bool = false,
            center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
    ) where {Nd, U <: AbstractArray{<:Complex}}
        ns = size(u)[1:Nd]
        xs = spatial_vectors(ns, ds; center = (-).(center))
        m = Nd == 2 ? f.(xs[1], xs[2]') : f.(xs[1])
        A = adapt_dim(U, Nd, eltype(m) <: Real ? real : complex)
        M = trainable ? Trainable : Static
        new{M, A}(A(m))
    end

    function Mask(u::ScalarField{U, Nd},
            ds::NTuple{Nd, Real},
            f::Function;
            trainable::Bool = false,
            center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
    ) where {Nd, U <: AbstractArray{<:Complex}}
        Mask(u.data, ds, f; trainable, center)
    end

    function Mask(u::ScalarField{U, Nd},
            f::Function;
            trainable::Bool = false,
            center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
    ) where {Nd, U <: AbstractArray{<:Complex}}
        Mask(u.data, u.ds, f; trainable, center)
    end
end

Functors.@functor Mask (m,)

Base.collect(p::Mask) = collect(p.m)
Base.size(p::Mask) = size(p.m)

function Base.fill!(p::Mask, v)
    p.m .= v
    p
end

function Base.fill!(p::Mask, v::AbstractArray)
    copyto!(p.m, v)
    p
end

trainable(p::Mask{<:Trainable}) = (; m = p.m)

function propagate(u::AbstractArray, p::Mask, direction::Type{<:Direction})
    u .* conj_direction(p.m, direction)
end

function propagate(u::ScalarField, p::Mask, direction::Type{<:Direction})
    data = u.data .* conj_direction(p.m, direction)
    ScalarField(data, u.ds, u.lambdas, u.lambdas_collection)
end
