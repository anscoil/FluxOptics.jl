struct Phase{M, A, U} <: AbstractCustomComponent{M}
    ϕ::A
    ∂p::Union{Nothing, @NamedTuple{ϕ::A}}
    u::Union{Nothing, U}

    function Phase(
            ϕ::A,
            ∂p::Union{Nothing, @NamedTuple{ϕ::A}},
            u::U
    ) where {T <: Real,
            A <: AbstractArray{T, 2}, U <: Union{Nothing, AbstractArray{Complex{T}}}}
        M = isnothing(u) ? Trainable{Unbuffered} : Trainable{Buffered}
        new{M, A, U}(ϕ, ∂p, u)
    end

    function Phase(
            u::U,
            ds::NTuple{Nd, Real},
            f::Function;
            trainable::Bool = false,
            buffered::Bool = false,
            center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
    ) where {N, Nd, U <: AbstractArray{<:Complex, N}}
        M = trainability(trainable, buffered)
        @assert Nd in (1, 2)
        @assert N >= Nd
        P = adapt_dim(U, Nd, real)
        xs = spatial_vectors(size(u)[1:Nd], ds; center = (-).(center))
        ϕ = Nd == 2 ? P(f.(xs[1], xs[2]')) : P(f.(xs[1]))
        ∂p = (trainable && buffered) ? (; ϕ = similar(ϕ)) : nothing
        u = (trainable && buffered) ? similar(u) : nothing
        A = typeof(ϕ)
        new{M, A, U}(ϕ, ∂p, u)
    end

    function Phase(
            u::ScalarField{U, Nd},
            ds::NTuple{Nd, Real},
            f::Function;
            trainable::Bool = false,
            buffered::Bool = false,
            center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
    ) where {U <: AbstractArray{<:Complex}, Nd}
        Phase(u.data, ds, f; trainable, buffered, center)
    end

    function Phase(
            u::ScalarField{U, Nd},
            f::Function;
            trainable::Bool = false,
            buffered::Bool = false,
            center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)
    ) where {U <: AbstractArray{<:Complex}, Nd}
        Phase(u.data, u.ds, f; trainable, buffered, center)
    end
end

Functors.@functor Phase (ϕ,)

Base.collect(p::Phase) = collect(p.ϕ)
Base.size(p::Phase) = size(p.ϕ)

function Base.fill!(p::Phase, v::Real)
    p.ϕ .= v
    p
end

function Base.fill!(p::Phase, v::AbstractArray)
    copyto!(p.ϕ, v)
    p
end

trainable(p::Phase{<:Trainable}) = (; ϕ = p.ϕ)

get_preallocated_gradient(p::Phase{Trainable{Buffered}}) = p.∂p

get_saved_buffer(p::Phase{Trainable{Buffered}}) = p.u

function apply_phase!(u::AbstractArray, p::Phase, ::Type{Forward})
    @. u *= cis(p.ϕ)
end

function apply_phase!(u::AbstractArray, p::Phase, ::Type{Backward})
    @. u *= cis(-p.ϕ)
end

function apply_phase!(u::ScalarField, p::Phase, direction::Type{<:Direction})
    apply_phase!(u.data, p, direction)
    u
end

function propagate!(u, p::Phase, direction::Type{<:Direction})
    apply_phase!(u, p, direction)
end

function backpropagate!(u, p::Phase, direction::Type{<:Direction})
    propagate!(u, p, reverse(direction))
end

function propagate_and_save!(u, p::Phase{<:Trainable}, direction::Type{<:Direction})
    copyto!(p.u, get_data(u))
    propagate!(u, p, direction)
end

function compute_phase_gradient!(
        ∂ϕ::P,
        ∂u::U,
        u::U) where {T <: Real, Nd,
        P <: AbstractArray{T, Nd},
        U <: AbstractArray{<:Complex{T}}}
    sdims = (Nd + 1):ndims(∂u)
    ∂ϕ .= sum(imag.(∂u .* conj.(u)), dims = sdims)
end

function compute_phase_gradient!(
        ∂ϕ::P,
        ∂u::U,
        u::U) where {T <: Real,
        P <: Array{T, 2},
        U <: Array{<:Complex{T}}}
    sdims = 3:ndims(∂u)
    ∂ϕ .= 0
    @inbounds for idx in CartesianIndices(size(∂u)[sdims])
        @inbounds for j in axes(∂ϕ, 2), i in axes(∂ϕ, 1)

            full_idx = (i, j, Tuple(idx)...)
            val = imag(∂u[full_idx...] * conj(u[full_idx...]))
            ∂ϕ[i, j] += val
        end
    end
    ∂ϕ
end

function backpropagate_with_gradient!(∂v, u_saved, ∂p::NamedTuple,
        p::Phase{<:Trainable}, direction::Type{<:Direction})
    ∂u = backpropagate!(∂v, p, direction)
    compute_phase_gradient!(∂p.ϕ, get_data(∂u), get_data(u_saved))
    (∂u, ∂p)
end
