struct Phase{T} <: AbstractOpticalComponent{T}
    ϕ
    ∇ϕ
    u

    function Phase(
            ϕ::A,
            ∇ϕ::A,
            u::U
    ) where {A <: AbstractArray{<:Real, 2}, U <: AbstractArray{<:Complex{}}}
        new{Trainable}(ϕ, ∇ϕ, u)
    end

    function Phase(ϕ::A, ∇ϕ::Nothing, u::Nothing) where {A <: AbstractArray{<:Real, 2}}
        new{Static}(ϕ, ∇ϕ, u)
    end

    function Phase(
            dims::NTuple{N, Integer},
            dx::Real,
            dy::Real,
            f::Function;
            U::Type{<:AbstractArray{<:Complex, N}} = Array{ComplexF64, N},
            trainability::Trainability = Static,
            xc::Real = 0,
            yc::Real = 0
    ) where {N}
        @assert N >= 2
        P = adapt_2D(U, real)
        nx, ny = dims
        xv, yv = spatial_vectors(nx, ny, dx, dy; xc = xc, yc = xc)
        ϕ = P(f.(xv, yv'))
        ∇ϕ = is_trainable(trainability) ? similar(ϕ) : nothing
        u = is_trainable(trainability) ? U(undef, dims) : nothing
        new{trainability}(ϕ, ∇ϕ, u)
    end

    function Phase(
            u::U,
            dx::Real,
            dy::Real,
            f::Function;
            trainability::Trainability = Static,
            xc::Real = 0,
            yc::Real = 0
    ) where {U <: AbstractArray{<:Complex}}
        Phase(size(u), dx, dy, f; U, trainable = trainable, xc = xc, yc = yc)
    end

    function Phase(u::U,
            ϕ::P;
            trainability::Trainability = Static
    ) where {N, U <: AbstractArray{<:Complex, N}, P <: AbstractArray{<:Real, 2}}
        @assert N >= 2
        ∇ϕ = is_trainable(trainability) ? similar(ϕ) : nothing
        u = is_trainable(trainability) ? similar(u) : nothing
        new{trainability}(ϕ, ∇ϕ, u)
    end

    function Phase(ϕ::A) where {A <: AbstractArray{<:Real, 2}}
        new{Static}(ϕ, nothing, nothing)
    end
end

trainable(p::Phase{Trainable}) = (; ϕ = p.ϕ)

function apply_phase!(u, p, ::Type{Forward})
    u .*= exp.(im .* p.ϕ)
end

function apply_phase!(u, p, ::Type{Backward})
    u .*= exp.(-im .* p.ϕ)
end

function propagate!(u, p::Phase, direction::Type{<:Direction})
    apply_phase!(u, p, direction)
end

function backpropagate!(u, p::Phase, direction::Type{<:Direction})
    propagate!(u, p, reverse(direction))
end

function propagate_and_save!(u, p::Phase{Trainable}, direction::Type{<:Direction})
    copyto!(p.u, u)
    apply_phase!(u, p, direction)
end

function compute_phase_gradient!(
        ∂ϕ::P,
        ∂u::U,
        u::U) where {T <: Real,
        P <: AbstractArray{T, 2},
        U <: AbstractArray{<:Complex}}
    nd = ndims(∂u)
    sdims = 3:nd

    @inbounds for j in axes(∂ϕ, 2), i in axes(∂ϕ, 1)

        acc = zero(T)
        for idx in size(∂u)[sdims]
            full_idx = (i, j, idx...)
            val = imag(∂u[full_idx...] * conj(u[full_idx...]))
            acc += val
        end
        ∂ϕ[i, j] = acc
    end
    ∂ϕ
end

function backpropagate_with_gradients!(∂v, ∂p::NamedTuple, p::Phase{Trainable},
        direction::Type{<:Direction})
    ∂u = backpropagate!(∂v, p, direction)
    compute_phase_gradient!(∂p.ϕ, ∂u, p.u)
    (∂u, ∂p)
end
