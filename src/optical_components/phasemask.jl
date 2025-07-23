struct Phase{M, A, U} <: AbstractOpticalComponent{M}
    ϕ::A
    ∂p::Union{Nothing, @NamedTuple{ϕ::A}}
    u::Union{Nothing, U}

    function Phase(
            ϕ::A,
            ∂p::@NamedTuple{ϕ::A},
            u::U
    ) where {A <: AbstractArray{<:Real, 2}, U <: AbstractArray{<:Complex}}
        @assert size(∂p.ϕ) == size(ϕ)
        @assert size(u)[1:2] == size(ϕ)
        new{Trainable{@NamedTuple{ϕ::A}}, A, U}(ϕ, ∂p, u)
    end

    function Phase(
            ϕ::A, u::U) where {A <: AbstractArray{<:Real, 2},
            U <: AbstractArray{<:Complex}}
        @assert size(u)[1:2] == size(ϕ)
        new{Trainable{Nothing}, A, U}(ϕ, nothing, u)
    end

    function Phase(ϕ::A) where {A <: AbstractArray{<:Real, 2}}
        new{Static, A, Nothing}(ϕ, nothing, nothing)
    end

    function Phase(ϕ::A, ∂p::Nothing,
            u::U) where {A <: AbstractArray{<:Real, 2}, U <: AbstractArray{<:Complex}}
        Phase(ϕ, u)
    end

    function Phase(ϕ::A, ∂p::Nothing, u::Nothing) where {A <: AbstractArray{<:Real, 2}}
        Phase(ϕ)
    end

    function Phase(
            U::Type{<:AbstractArray{<:Complex, N}},
            dims::NTuple{N, Integer},
            dx::Real,
            dy::Real,
            f::Function;
            trainable::Bool = false,
            prealloc_gradient::Bool = false,
            xc::Real = 0,
            yc::Real = 0
    ) where {N}
        @assert N >= 2
        P = adapt_dim(U, 2, real)
        nx, ny = dims
        xv, yv = spatial_vectors(nx, ny, dx, dy; xc = xc, yc = xc)
        ϕ = P(f.(xv, yv'))
        ∂p = prealloc_gradient ? (; ϕ = similar(ϕ)) : nothing
        u = trainable ? U(undef, dims) : nothing
        Phase(ϕ, ∂p, u)
    end

    function Phase(
            u::U,
            dx::Real,
            dy::Real,
            f::Function;
            trainable::Bool = false,
            prealloc_gradient::Bool = false,
            xc::Real = 0,
            yc::Real = 0
    ) where {U <: AbstractArray{<:Complex}}
        Phase(U, size(u), dx, dy, f, trainable = trainable,
            prealloc_gradient = prealloc_gradient, xc = xc, yc = yc)
    end
end

trainable(p::Phase{<:Trainable}) = (; ϕ = p.ϕ)

get_preallocated_gradient(p::Phase{<:Trainable{<:NamedTuple}}) = p.∂p

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

function propagate_and_save!(u, p::Phase{<:Trainable}, direction::Type{<:Direction})
    copyto!(p.u, u)
    apply_phase!(u, p, direction)
end

function compute_phase_gradient!(
        ∂ϕ::P,
        ∂u::U,
        u::U) where {T <: Real,
        P <: AbstractArray{T, 2},
        U <: AbstractArray{<:Complex{T}}}
    sdims = 3:ndims(∂u)
    @views ∂ϕ .= sum(imag.(∂u .* conj.(u)), dims = sdims)
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

function backpropagate_with_gradient!(∂v, ∂p::NamedTuple, p::Phase{<:Trainable},
        direction::Type{<:Direction})
    ∂u = backpropagate!(∂v, p, direction)
    compute_phase_gradient!(∂p.ϕ, ∂u, p.u)
    (∂u, ∂p)
end
