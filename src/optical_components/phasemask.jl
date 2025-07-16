struct Phase{T} <: AbstractOpticalComponent{T}
    ϕ
    ∇ϕ
    u_fwd

    function Phase(ϕ::A, ∇ϕ::A, u_fwd) where {A <: AbstractArray{<:Real, 2}}
        new{Trainable}(ϕ, ∇ϕ, u_fwd)
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
        u_fwd = is_trainable(trainability) ? U(undef, dims) : nothing
        new{trainability}(ϕ, ∇ϕ, u_fwd)
    end

    function Phase(
            u::AbstractArray{<:Complex{<:Real}},
            dx::Real,
            dy::Real,
            f::Function;
            trainability::Trainability = Static,
            xc::Real = 0,
            yc::Real = 0
    )
        Phase(size(u), dx, dy, f; U = typeof(u), trainable = trainable, xc = xc, yc = yc)
    end

    function Phase(
            u::AbstractArray{<:Complex{<:Real}, N},
            ϕ::P;
            trainability::Trainability = Static
    ) where {N, P <: AbstractArray{<:Real, 2}}
        @assert N >= 2
        ∇ϕ = is_trainable(trainability) ? similar(ϕ) : nothing
        u_fwd = is_trainable(trainability) ? similar(u) : nothing
        new{trainability}(ϕ, ∇ϕ, u_fwd)
    end

    function Phase(ϕ::AbstractArray{<:Real, 2})
        new{Static}(ϕ, nothing, nothing)
    end
end

function apply_phase(u, phi, ::Type{Forward})
    u .*= exp.(im .* phi.ϕ)
end

function apply_phase(u, phi, ::Type{Backward})
    u .*= exp.(-im .* phi.ϕ)
end

function propagate!(u, phi::Phase{Static};
        direction::Type{<:Direction} = Forward)
    apply_phase(u, phi, direction)
end

function propagate!(u, P::Phase{Trainable};
        direction::Type{<:Direction} = Forward,
        save_u::Bool = false)
    if save_u
        copyto!(P.u_fwd, u)
    end
    apply_phase(u, P, direction)
end
