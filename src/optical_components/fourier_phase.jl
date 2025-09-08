struct FourierPhase{M, A, B, U, P} <: AbstractCustomComponent{M}
    ϕ::A
    aperture::B
    p_f::P
    ∂p::Union{Nothing, @NamedTuple{ϕ::A}}
    u::Union{Nothing, U}

    function FourierPhase(ϕ::A, aperture::B, p_f::P,
            ∂p::Union{Nothing, @NamedTuple{m::A}}, u::U) where {A, B, U, P}
        M = isnothing(u) ? Trainable{Unbuffered} : Trainable{Buffered}
        new{M, A, B, U, P}(ϕ, aperture, p_f, ∂p, u)
    end

    function FourierPhase(u::ScalarField{U, Nd},
            ds::NTuple{Nd, Real},
            phase::AbstractArray{<:Real},
            aperture::AbstractArray{Bool} = [true];
            trainable::Bool = false, buffered::Bool = false
    ) where {Nd, U <: AbstractArray{<:Complex}}
        M = trainability(trainable, buffered)
        ns = size(u.data)[1:Nd]
        u_plan = similar(u.data)
        p_f = make_fft_plans(u_plan, Tuple(1:Nd))
        P = typeof(p_f)
        @assert isbroadcastable(phase, u)
        @assert isbroadcastable(aperture, u)
        A = adapt_dim(U, ndims(phase), real)
        B = adapt_dim(U, ndims(aperture), real)
        ϕ = A(phase)
        aperture = B(aperture)
        ∂p = (trainable && buffered) ? (; ϕ = similar(ϕ)) : nothing
        u = (trainable && buffered) ? u_plan : nothing
        new{M, A, B, U, P}(ϕ, aperture, p_f, ∂p, u)
    end

    function FourierPhase(u::ScalarField{U, Nd},
            phase::AbstractArray{<:Real},
            aperture::AbstractArray{Bool} = [true];
            trainable::Bool = false, buffered::Bool = false
    ) where {Nd, U <: AbstractArray{<:Complex}}
        FourierPhase(u, u.ds, phase, aperture; trainable, buffered)
    end
end

Functors.@functor FourierPhase (ϕ,)

get_data(p::FourierPhase) = p.ϕ

trainable(p::FourierPhase{<:Trainable}) = (; ϕ = p.ϕ)

get_preallocated_gradient(p::FourierPhase{Trainable{Buffered}}) = p.∂p

alloc_saved_buffer(u::ScalarField, p::FourierPhase{Trainable{Unbuffered}}) = similar(u.data)

get_saved_buffer(p::FourierPhase{Trainable{Buffered}}) = p.u

function propagate!(u::ScalarField, p::FourierPhase, direction::Type{<:Direction};
        u_saved = nothing)
    p.p_f.ft * u.data
    copyto!(u_saved, u.data)
    s = sign(direction)
    @. u.data *= p.aperture*cis(s*p.ϕ)
    p.p_f.ift * u.data
    u
end

function propagate_and_save!(u::ScalarField, p::FourierPhase{Trainable{Buffered}},
        direction::Type{<:Direction})
    propagate!(u, p, direction; u_saved = p.u)
end

function propagate_and_save!(u::ScalarField, u_saved::AbstractArray,
        p::FourierPhase{Trainable{Unbuffered}}, direction::Type{<:Direction})
    propagate!(u, p, direction; u_saved)
end

compute_phase_gradient!(::Nothing, ::Nothing, ∂u, direction) = nothing

function backpropagate!(u::ScalarField, p::FourierPhase, direction::Type{<:Direction};
        u_saved = nothing, ∂p = nothing)
    ∂ϕ = isnothing(∂p) ? nothing : ∂p.ϕ
    p.p_f.ft * u.data
    s = sign(reverse(direction))
    @. u.data *= p.aperture*cis(s*p.ϕ)
    compute_phase_gradient!(∂ϕ, u_saved, u, direction)
    p.p_f.ift * u.data
    u
end

function backpropagate_with_gradient!(∂v::ScalarField, u_saved::AbstractArray,
        ∂p::NamedTuple, p::FourierPhase{<:Trainable}, direction::Type{<:Direction})
    ∂u = backpropagate!(∂v, p, direction; u_saved, ∂p)
    (∂u, ∂p)
end
