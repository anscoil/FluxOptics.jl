struct TeaDOE{M, T, A, U} <: AbstractOpticalComponent{M}
    dn::T
    r::Complex{T}
    h::A
    ∂p::Union{Nothing, @NamedTuple{h::A}}
    u::Union{Nothing, U}

    function TeaDOE(
            dn::T,
            r::Complex{T},
            h::A,
            ∂p::@NamedTuple{h::A},
            u::U;
    ) where {T <: Real, A <: AbstractArray{T, 2}, U <: AbstractArray{Complex{T}}}
        @assert size(∂p.h) == size(h)
        @assert size(u)[1:ndims(h)] == size(h)
        new{Trainable{GradAllocated}, T, A, U}(dn, r, h, ∂p, u)
    end

    function TeaDOE(
            dn::T, r::Complex{T}, h::A, u::U
    ) where {T <: Real,
             A <: AbstractArray{T, 2},
             U <: AbstractArray{Complex{T}}}
        @assert size(u)[1:ndims(h)] == size(h)
        new{Trainable{GradNoAlloc}, T, A, U}(dn, r, h, nothing, u)
    end

    function TeaDOE(dn::T, r::Complex{T}, h::A
    ) where {T <: Real, A <: AbstractArray{T, 2}}
        new{Static, T, A, Nothing}(dn, r, h, nothing, nothing)
    end

    function TeaDOE(dn::T,
            r::Complex{T},
            h::A,
            ∂p::Nothing,
            u::U
    ) where {T <: Real, A <: AbstractArray{T, 2}, U <: AbstractArray{<:Complex}}
        TeaDOE(dn, r, h, u)
    end

    function TeaDOE(dn::T, r::Complex{T}, h::A, ∂p::Nothing, u::Nothing
    ) where {T <: Real, A <: AbstractArray{T, 2}}
        TeaDOE(dn, r, h)
    end

    function TeaDOE(
            U::Type{<:AbstractArray{Complex{T}, N}},
            dims::NTuple{N, Integer},
            ds::NTuple{Nd, Real},
            dn::Real,
            f::Function;
            r::Number = 1,
            trainable::Bool = false,
            prealloc_gradient::Bool = false,
            center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
    ) where {T <: Real, N, Nd}
        @assert Nd in (1, 2)
        @assert N >= Nd
        P = adapt_dim(U, Nd, real)
        xs = spatial_vectors(dims[1:Nd], ds; center = center)
        h = Nd == 2 ? P(f.(xs[1], xs[2]')) : P(f.(xs[1]))
        ∂p = prealloc_gradient ? (; h = similar(h)) : nothing
        u = trainable ? U(undef, dims) : nothing
        TeaDOE(T(dn), Complex{T}(r), h, ∂p, u)
    end

    function TeaDOE(
            u::Union{U, ScalarField{U}},
            ds::NTuple{Nd, Real},
            dn::Real,
            f::Function;
            r::Number = 1,
            trainable::Bool = false,
            prealloc_gradient::Bool = false,
            center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
    ) where {U <: AbstractArray{<:Complex}, Nd}
        TeaDOE(U, size(u), ds, dn, f; r = r, trainable = trainable,
            prealloc_gradient = prealloc_gradient, center = center)
    end
end

function TeaReflector(
        u::Union{U, ScalarField{U}},
        ds::NTuple{Nd, Real},
        f::Function;
        r::Number = 1,
        trainable::Bool = false,
        prealloc_gradient::Bool = false,
        center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
) where {U <: AbstractArray{<:Complex}, Nd}
    TeaDOE(U, size(u), ds, 2, f; r = r, trainable = trainable,
        prealloc_gradient = prealloc_gradient, center = center)
end

trainable(p::TeaDOE{<:Trainable}) = (; h = p.h)

get_preallocated_gradient(p::TeaDOE{<:Trainable{GradAllocated}}) = p.∂p

function apply_phase!(
        u::AbstractArray, lambdas, p::TeaDOE{M, T}, ::Type{Forward}) where {M, T}
    @. u *= p.r * cis((T(2)*π/lambdas)*p.dn*p.h)
end

function apply_phase!(
        u::AbstractArray, lambdas, p::TeaDOE{M, T}, ::Type{Backward}) where {M, T}
    @. u *= conj(p.r) * cis(-(T(2)*π/lambdas)*p.dn*p.h)
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
        lambdas,
        dn::T,
        r::Complex{T},
        u::U) where {T <: Real,
        Nd,
        P <: AbstractArray{T, Nd},
        U <: AbstractArray{<:Complex{T}}}
    sdims = (Nd + 1):ndims(∂u)
    g = @. (T(2)*π*dn/lambdas)*imag(∂u * conj(r*u))
    @views ∂h .= sum(g, dims = sdims)
end

function backpropagate_with_gradient!(∂v::ScalarField, ∂p::NamedTuple,
        p::TeaDOE{<:Trainable}, direction::Type{<:Direction})
    ∂u = backpropagate!(∂v, p, direction)
    compute_surface_gradient!(∂p.h, ∂u.data, ∂u.lambdas, p.dn, p.r, p.u)
    (∂u, ∂p)
end
