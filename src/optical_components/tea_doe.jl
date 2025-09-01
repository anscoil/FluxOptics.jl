struct TeaDOE{M, Fn, Fr, A, U} <: AbstractCustomComponent{M}
    dn::Fn
    r::Fr
    h::A
    ∂p::Union{Nothing, @NamedTuple{h::A}}
    u::Union{Nothing, U}

    function TeaDOE(
            dn::Fn,
            r::Fr,
            h::A,
            ∂p::@NamedTuple{h::A},
            u::U;
    ) where {Fn <: Function, Fr <: Function, T <: Real,
            A <: AbstractArray{T, 2}, U <: AbstractArray{Complex{T}}}
        @assert size(∂p.h) == size(h)
        @assert size(u)[1:ndims(h)] == size(h)
        new{Trainable{GradAllocated}, Fn, Fr, A, U}(dn, r, h, ∂p, u)
    end

    function TeaDOE(
            dn::Fn, r::Fr,
            h::A,
            u::U
    ) where {Fn <: Function, Fr <: Function, T <: Real,
            A <: AbstractArray{T, 2},
            U <: AbstractArray{Complex{T}}}
        @assert size(u)[1:ndims(h)] == size(h)
        new{Trainable{GradNoAlloc}, Fn, Fr, A, U}(dn, r, h, nothing, u)
    end

    function TeaDOE(dn::Fn,
            r::Fr,
            h::A
    ) where {Fn <: Function, Fr <: Function, T <: Real, A <: AbstractArray{T, 2}}
        new{Static, Fn, Fr, A, Nothing}(dn, r, h, nothing, nothing)
    end

    function TeaDOE(dn::Fn,
            r::Fr,
            h::A,
            ∂p::Nothing,
            u::U
    ) where {Fn <: Function, Fr <: Function, T <: Real,
            A <: AbstractArray{T, 2}, U <: AbstractArray{<:Complex}}
        TeaDOE(dn, r, h, u)
    end

    function TeaDOE(dn::Fn,
            r::Fr,
            h::A,
            ∂p::Nothing,
            u::Nothing
    ) where {Fn <: Function, Fr <: Function, T <: Real, A <: AbstractArray{T, 2}}
        TeaDOE(dn, r, h)
    end

    function TeaDOE(
            u::U,
            ds::NTuple{Nd, Real},
            dn::Union{Real, Function},
            f::Function;
            r::Union{Number, Function} = 1,
            trainable::Bool = false,
            prealloc_gradient::Bool = false,
            center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
    ) where {N, Nd, T, U <: AbstractArray{Complex{T}, N}}
        check_trainable_combination(trainable, prealloc_gradient)
        @assert Nd in (1, 2)
        @assert N >= Nd
        P = adapt_dim(U, Nd, real)
        xs = spatial_vectors(size(u)[1:Nd], ds; center = (-).(center))
        h = Nd == 2 ? P(f.(xs[1], xs[2]')) : P(f.(xs[1]))
        ∂p = prealloc_gradient ? (; h = similar(h)) : nothing
        u = trainable ? similar(u) : nothing
        dn_f = isa(dn, Real) ? (λ -> T(dn)) : (λ -> T(dn(λ)))
        r_f = isa(r, Number) ? (λ -> Complex{T}(r)) : (λ -> Complex{T}(r(λ)))
        TeaDOE(dn_f, r_f, h, ∂p, u)
    end

    function TeaDOE(
            u::ScalarField{U, Nd},
            dn::Union{Real, Function},
            f::Function;
            r::Union{Number, Function} = 1,
            trainable::Bool = false,
            prealloc_gradient::Bool = false,
            center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
    ) where {U <: AbstractArray{<:Complex}, Nd}
        TeaDOE(u.data, u.ds, dn, f; r = r, trainable = trainable,
            prealloc_gradient = prealloc_gradient, center = center)
    end
end

function TeaReflector(
        u::ScalarField{U, Nd},
        f::Function;
        r::Union{Number, Function} = 1,
        trainable::Bool = false,
        prealloc_gradient::Bool = false,
        center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
) where {U <: AbstractArray{<:Complex}, Nd}
    TeaDOE(u, λ -> 2, f; r = r, trainable = trainable,
        prealloc_gradient = prealloc_gradient, center = center)
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

get_preallocated_gradient(p::TeaDOE{Trainable{GradAllocated}}) = p.∂p

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
    g = @. (T(2)*π*dn(lambdas)/lambdas)*imag(∂u * conj(r(lambdas)*u))
    copyto!(∂h, sum(g; dims = sdims))
end

function backpropagate_with_gradient!(∂v::ScalarField, ∂p::NamedTuple,
        p::TeaDOE{<:Trainable}, direction::Type{<:Direction})
    ∂u = backpropagate!(∂v, p, direction)
    compute_surface_gradient!(∂p.h, ∂u.data, p.u, ∂u.lambdas, p.dn, p.r)
    (∂u, ∂p)
end
