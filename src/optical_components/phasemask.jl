struct Phase{M, A, U} <: AbstractCustomComponent{M}
    ϕ::A
    ∂p::Union{Nothing, @NamedTuple{ϕ::A}}
    u::Union{Nothing, U}

    function Phase(ϕ::A, ∂p::Union{Nothing, @NamedTuple{ϕ::A}}, u::U) where {A, U}
        M = isnothing(u) ? Trainable{Unbuffered} : Trainable{Buffered}
        new{M, A, U}(ϕ, ∂p, u)
    end

    function Phase(
            u::ScalarField{U, Nd},
            ds::NTuple{Nd, Real},
            f::Function = (x...) -> 0;
            trainable::Bool = false,
            buffered::Bool = false
    ) where {N, Nd, U <: AbstractArray{<:Complex, N}}
        M = trainability(trainable, buffered)
        @assert Nd in (1, 2)
        @assert N >= Nd
        P = adapt_dim(U, Nd, real)
        xs = spatial_vectors(size(u.data)[1:Nd], ds)
        ϕ = Nd == 2 ? P(f.(xs[1], xs[2]')) : P(f.(xs[1]))
        ∂p = (trainable && buffered) ? (; ϕ = similar(ϕ)) : nothing
        u = (trainable && buffered) ? similar(u.data) : nothing
        A = typeof(ϕ)
        new{M, A, U}(ϕ, ∂p, u)
    end

    function Phase(
            u::ScalarField{U, Nd},
            f::Function = (x...) -> 0;
            trainable::Bool = false,
            buffered::Bool = false
    ) where {U <: AbstractArray{<:Complex}, Nd}
        Phase(u, u.ds, f; trainable, buffered)
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

alloc_saved_buffer(u::ScalarField, p::Phase{Trainable{Unbuffered}}) = similar(u.data)

get_saved_buffer(p::Phase{Trainable{Buffered}}) = p.u

function propagate!(u::ScalarField, p::Phase, direction::Type{<:Direction})
    s = sign(direction)
    @. u.data *= cis(s*p.ϕ)
    u
end

function backpropagate!(u::ScalarField, p::Phase, direction::Type{<:Direction})
    propagate!(u, p, reverse(direction))
end

function propagate_and_save!(u::ScalarField, p::Phase{Trainable{Buffered}},
        direction::Type{<:Direction})
    copyto!(p.u, u.data)
    propagate!(u, p, direction)
end

function propagate_and_save!(u::ScalarField, u_saved::AbstractArray,
        p::Phase{Trainable{Unbuffered}}, direction::Type{<:Direction})
    copyto!(u_saved, u.data)
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

function backpropagate_with_gradient!(∂v::ScalarField, u_saved::AbstractArray,
        ∂p::NamedTuple, p::Phase{<:Trainable}, direction::Type{<:Direction})
    ∂u = backpropagate!(∂v, p, direction)
    compute_phase_gradient!(∂p.ϕ, ∂u.data, get_data(u_saved))
    ∂p.ϕ .*= sign(direction)
    (∂u, ∂p)
end
