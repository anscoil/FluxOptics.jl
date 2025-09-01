struct ComplexMask{M, A} <: AbstractPureComponent{M}
    m::A

    function ComplexMask(m::A) where {A}
        new{Static, A}(m)
    end

    function ComplexMask(u::U,
            ds::NTuple{Nd, Real},
            f::Function;
            trainable::Bool = false,
            center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
    ) where {Nd, U <: AbstractArray{<:Complex}}
        ns = size(u)[1:Nd]
        A = adapt_dim(U, Nd)
        xs = spatial_vectors(ns, ds; center = (-).(center))
        m = Nd == 2 ? A(f.(xs[1], xs[2]')) : A(f.(xs[1]))
        M = trainable ? Trainable{GradNoAlloc} : Static
        new{M, A}(m)
    end

    function ComplexMask(u::ScalarField{U, Nd},
            f::Function;
            trainable::Bool = false,
            center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
    ) where {Nd, U <: AbstractArray{<:Complex}}
        ComplexMask(u.data, u.ds, f; trainable = trainable, center = center)
    end
end

Functors.@functor ComplexMask (m,)

Base.collect(p::ComplexMask) = collect(p.m)
Base.size(p::ComplexMask) = size(p.m)

function Base.fill!(p::ComplexMask, v::Real)
    p.m .= v
    p
end

function Base.fill!(p::ComplexMask, v::AbstractArray)
    copyto!(p.m, v)
    p
end

trainable(p::ComplexMask{<:Trainable}) = (; m = p.m)

function propagate(u::AbstractArray, p::ComplexMask, direction::Type{<:Direction})
    u .* conj_direction(p.m, direction)
end

function propagate(u::ScalarField, p::ComplexMask, direction::Type{<:Direction})
    data = u.data .* conj_direction(p.m, direction)
    ScalarField(data, u.ds, u.lambdas, u.lambdas_collection)
end
