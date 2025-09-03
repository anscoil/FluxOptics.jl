struct TeaDOE{M, Fn, Fr, A, U} <: AbstractCustomComponent{M}
    dn::Fn
    r::Fr
    h::A
    ∂p::Union{Nothing, @NamedTuple{h::A}}
    u::Union{Nothing, U}

    function TeaDOE(dn::Fn, r::Fr, h::A,
            ∂p::Union{Nothing, @NamedTuple{h::A}},
            u::U) where {Fn, Fr, A, U}
        M = isnothing(u) ? Trainable{Unbuffered} : Trainable{Buffered}
        new{M, Fn, Fr, A, U}(dn, r, h, ∂p, u)
    end

    function TeaDOE(
            u::U,
            ds::NTuple{Nd, Real},
            dn::Union{Real, Function},
            f::Function;
            r::Union{Number, Function} = 1,
            trainable::Bool = false,
            buffered::Bool = false,
            center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
    ) where {N, Nd, T, U <: AbstractArray{Complex{T}, N}}
        @assert Nd in (1, 2)
        @assert N >= Nd
        M = trainability(trainable, buffered)
        P = adapt_dim(U, Nd, real)
        xs = spatial_vectors(size(u)[1:Nd], ds; center = (-).(center))
        h = Nd == 2 ? P(f.(xs[1], xs[2]')) : P(f.(xs[1]))
        ∂p = (trainable && buffered) ? (; h = similar(h)) : nothing
        u = (trainable && buffered) ? similar(u) : nothing
        dn_f = isa(dn, Real) ? (λ -> T(dn)) : (λ -> T(dn(λ)))
        r_f = isa(r, Number) ? (λ -> Complex{T}(r)) : (λ -> Complex{T}(r(λ)))
        Fn = typeof(dn_f)
        Fr = typeof(r_f)
        A = typeof(h)
        new{M, Fn, Fr, A, U}(dn_f, r_f, h, ∂p, u)
    end

    function TeaDOE(
            u::ScalarField{U, Nd},
            ds::NTuple{Nd, Real},
            dn::Union{Real, Function},
            f::Function;
            r::Union{Number, Function} = 1,
            trainable::Bool = false,
            buffered::Bool = false,
            center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
    ) where {U <: AbstractArray{<:Complex}, Nd}
        TeaDOE(u.data, ds, dn, f; r, trainable, buffered, center)
    end

    function TeaDOE(
            u::ScalarField{U, Nd},
            dn::Union{Real, Function},
            f::Function;
            r::Union{Number, Function} = 1,
            trainable::Bool = false,
            buffered::Bool = false,
            center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
    ) where {U <: AbstractArray{<:Complex}, Nd}
        TeaDOE(u.data, u.ds, dn, f; r, trainable, buffered, center)
    end
end

function TeaReflector(
        u::ScalarField{U, Nd},
        ds::NTuple{Nd, Real},
        f::Function;
        r::Union{Number, Function} = 1,
        trainable::Bool = false,
        buffered::Bool = false,
        center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
) where {U <: AbstractArray{<:Complex}, Nd}
    TeaDOE(u, ds, 2, f; r, trainable, buffered, center)
end

function TeaReflector(
        u::ScalarField{U, Nd},
        f::Function;
        r::Union{Number, Function} = 1,
        trainable::Bool = false,
        buffered::Bool = false,
        center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
) where {U <: AbstractArray{<:Complex}, Nd}
    TeaDOE(u, 2, f; r, trainable, buffered, center)
end

Functors.@functor TeaDOE (h,)

Base.collect(p::TeaDOE) = collect(p.h)
Base.size(p::TeaDOE) = size(p.h)

function Base.fill!(p::TeaDOE, v::Real)
    p.h .= v
    p
end

function Base.fill!(p::TeaDOE, v::AbstractArray)
    copyto!(p.h, v)
    p
end

trainable(p::TeaDOE{<:Trainable}) = (; h = p.h)

get_preallocated_gradient(p::TeaDOE{Trainable{Buffered}}) = p.∂p

get_saved_buffer(p::TeaDOE{Trainable{Buffered}}) = p.u

function apply_phase!(
        u::AbstractArray{T}, lambdas, p::TeaDOE, ::Type{Forward}) where {T}
    @. u *= p.r(lambdas) * cis((T(2)*π/lambdas)*p.dn(lambdas)*p.h)
end

function apply_phase!(
        u::AbstractArray{T}, lambdas, p::TeaDOE, ::Type{Backward}) where {T}
    @. u *= conj(p.r(lambdas)) * cis(-(T(2)*π/lambdas)*p.dn(lambdas)*p.h)
end

function propagate!(u::AbstractArray, lambdas, p::TeaDOE, direction::Type{<:Direction})
    apply_phase!(u, lambdas, p, direction)
end

function propagate!(u::ScalarField, p::TeaDOE, direction::Type{<:Direction})
    apply_phase!(u.data, u.lambdas, p, direction)
    u
end

function backpropagate!(u::ScalarField, p::TeaDOE, direction::Type{<:Direction})
    propagate!(u, p, reverse(direction))
end

function propagate_and_save!(
        u::ScalarField, p::TeaDOE{<:Trainable}, direction::Type{<:Direction})
    copyto!(p.u, u.data)
    propagate!(u, p, direction)
end

function compute_surface_gradient!(
        ∂h::P,
        ∂u::U,
        u::U,
        lambdas,
        dn::Function,
        r::Function) where {T <: Real,
        Nd,
        P <: AbstractArray{T, Nd},
        U <: AbstractArray{<:Complex{T}}}
    sdims = (Nd + 1):ndims(∂u)
    g = @. (T(2)*π*dn(lambdas)/lambdas)*imag(∂u*conj(u))
    copyto!(∂h, sum(g; dims = sdims))
end

function backpropagate_with_gradient!(∂v::ScalarField, u_saved,
        ∂p::NamedTuple, p::TeaDOE{<:Trainable}, direction::Type{<:Direction})
    ∂u = backpropagate!(∂v, p, direction)
    compute_surface_gradient!(∂p.h, ∂u.data, get_data(u_saved), ∂u.lambdas, p.dn, p.r)
    (∂u, ∂p)
end
