struct FourierMask{M, A, U, P} <: AbstractCustomComponent{M}
    m::A
    p_f::P
    ∂p::Union{Nothing, @NamedTuple{m::A}}
    u::Union{Nothing, U}

    function FourierMask(m::A, p_f::P,
            ∂p::Union{Nothing, @NamedTuple{m::A}}, u::U) where {A, U, P}
        M = isnothing(u) ? Trainable{Unbuffered} : Trainable{Buffered}
        new{M, A, U, P}(m, p_f, ∂p, u)
    end

    function FourierMask(u::ScalarField{U, Nd},
            ds::NTuple{Nd, Real},
            f::Union{Function, AbstractArray} = (_...) -> 1;
            trainable::Bool = false, buffered::Bool = false
    ) where {Nd, U <: AbstractArray{<:Complex}}
        M = trainability(trainable, buffered)
        ns = size(u.data)[1:Nd]
        u_plan = similar(u.data)
        p_f = make_fft_plans(u_plan, Tuple(1:Nd))
        P = typeof(p_f)
        if isa(f, Function)
            A = adapt_dim(U, Nd)
            fs = [fftfreq(nx, 1/dx) for (nx, dx) in zip(ns, ds)]
            m = Nd == 2 ? A(f.(fs[1], fs[2]')) : A(f.(fs[1]))
        else
            @assert isbroadcastable(f, u)
            A = adapt_dim(U, ndims(f))
            m = A(f)
        end
        ∂p = (trainable && buffered) ? (; m = similar(m)) : nothing
        u = (trainable && buffered) ? u_plan : nothing
        new{M, A, U, P}(m, p_f, ∂p, u)
    end

    function FourierMask(u::ScalarField{U, Nd},
            f::Union{Function, AbstractArray} = (_...) -> 1;
            trainable::Bool = false, buffered::Bool = false
    ) where {Nd, U <: AbstractArray{<:Complex}}
        FourierMask(u, u.ds, f; trainable, buffered)
    end
end

Functors.@functor FourierMask (m,)

get_data(p::FourierMask) = p.m

trainable(p::FourierMask{<:Trainable}) = (; m = p.m)

get_preallocated_gradient(p::FourierMask{Trainable{Buffered}}) = p.∂p

alloc_saved_buffer(u::ScalarField, p::FourierMask{Trainable{Unbuffered}}) = similar(u.data)

get_saved_buffer(p::FourierMask{Trainable{Buffered}}) = p.u

function propagate!(u::ScalarField, p::FourierMask, direction::Type{<:Direction};
        u_saved = nothing)
    p.p_f.ft * u.data
    copyto!(u_saved, u.data)
    @. u.data *= conj_direction(p.m, direction)
    p.p_f.ift * u.data
    u
end

function propagate_and_save!(u::ScalarField, p::FourierMask{Trainable{Buffered}},
        direction::Type{<:Direction})
    propagate!(u, p, direction; u_saved = p.u)
end

function propagate_and_save!(u::ScalarField, u_saved::AbstractArray,
        p::FourierMask{Trainable{Unbuffered}}, direction::Type{<:Direction})
    propagate!(u, p, direction; u_saved)
end

function compute_mask_gradient!(∂m::AbstractArray{<:Complex, Nd}, u_saved, ∂u::ScalarField,
        direction) where {Nd}
    sdims = (Nd + 1):ndims(∂u.data)
    g = @. conj_direction(∂u.data*conj(u_saved), direction)
    copyto!(∂m, sum(g; dims = sdims))
end

compute_mask_gradient!(::Nothing, ::Nothing, ∂u, direction) = nothing

function backpropagate!(u::ScalarField, p::FourierMask, direction::Type{<:Direction};
        u_saved = nothing, ∂p = nothing)
    ∂m = isnothing(∂p) ? nothing : ∂p.m
    p.p_f.ft * u.data
    @. u.data *= conj_direction(p.m, reverse(direction))
    compute_mask_gradient!(∂m, u_saved, u, direction)
    p.p_f.ift * u.data
    u
end

function backpropagate_with_gradient!(∂v::ScalarField, u_saved::AbstractArray,
        ∂p::NamedTuple, p::FourierMask{<:Trainable}, direction::Type{<:Direction})
    ∂u = backpropagate!(∂v, p, direction; u_saved, ∂p)
    (∂u, ∂p)
end
