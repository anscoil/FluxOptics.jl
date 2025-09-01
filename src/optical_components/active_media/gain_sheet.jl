struct GainSheet{M, T, A} <: AbstractPureComponent{M}
    g0::A
    dz::T
    Isat::T

    function GainSheet(g0::A, dz::T, Isat::T) where {T, A}
        new{Trainable, T, A}(g0, dz, Isat)
    end

    function GainSheet(u::U,
            ds::NTuple{Nd, Real},
            dz::Real,
            Isat::Real,
            f::Function;
            trainable::Bool = false,
            center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
    ) where {Nd, T, U <: AbstractArray{Complex{T}}}
        ns = size(u)[1:Nd]
        A = adapt_dim(U, Nd, real)
        xs = spatial_vectors(ns, ds; center = (-).(center))
        g0 = Nd == 2 ? A(f.(xs[1], xs[2]')) : A(f.(xs[1]))
        M = trainable ? Trainable : Static
        new{M, T, A}(g0, dz, Isat)
    end

    function GainSheet(u::ScalarField{U, Nd},
            dz::Real,
            Isat::Real,
            f::Function;
            trainable::Bool = false,
            center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
    ) where {Nd, U <: AbstractArray{<:Complex}}
        GainSheet(u.data, u.ds, dz, Isat, f; trainable = trainable, center = center)
    end
end

Functors.@functor GainSheet (g0,)

Base.collect(p::GainSheet) = collect(p.g0)
Base.size(p::GainSheet) = size(p.g0)

function Base.fill!(p::GainSheet, v::Real)
    p.g0 .= v
    p
end

function Base.fill!(p::GainSheet, v::AbstractArray)
    copyto!(p.g0, v)
    p
end

trainable(p::GainSheet{<:Trainable}) = (; g0 = p.g0)

function propagate(u::AbstractArray, p::GainSheet, ::Type{<:Direction})
    dims = Tuple(1:ndims(p.g0))
    u .* exp.((p.g0*p.dz) ./ (1 .+ sum(abs2, u; dims = dims)/p.Isat))
end

function propagate(u::ScalarField, p::GainSheet, ::Type{<:Direction})
    data = u.data .* exp.((p.g0*p.dz) ./ (1 .+ intensity(u)/p.Isat))
    ScalarField(data, u.ds, u.lambdas, u.lambdas_collection)
end
