struct GainSheet{M, T, A} <: AbstractPureComponent{M}
    g0::A
    dz::T
    Isat::T

    function GainSheet(g0::A, dz::T, Isat::T) where {T, A}
        new{Trainable, T, A}(g0, dz, Isat)
    end

    function GainSheet(u::ScalarField{U, Nd},
            ds::NTuple{Nd, Real},
            dz::Real,
            Isat::Real,
            f::Function;
            trainable::Bool = false
    ) where {Nd, T, U <: AbstractArray{Complex{T}}}
        ns = size(u.data)[1:Nd]
        A = similar(U, real, Nd)
        xs = spatial_vectors(ns, ds)
        g0 = Nd == 2 ? A(f.(xs[1], xs[2]')) : A(f.(xs[1]))
        M = trainable ? Trainable : Static
        new{M, T, A}(g0, dz, Isat)
    end
end

Functors.@functor GainSheet (g0,)

get_data(p::GainSheet) = p.g0

trainable(p::GainSheet{<:Trainable}) = (; g0 = p.g0)

function propagate(u::ScalarField, p::GainSheet, ::Type{<:Direction})
    data = u.data .* exp.((p.g0*p.dz) ./ (1 .+ intensity(u)/p.Isat))
    set_field_data(u, data)
end
