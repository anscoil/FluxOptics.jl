struct FourierWrapper{M, S, W, P} <: AbstractPureComponent{M}
    wrapped_components::W
    p_f::P
    s::S

    function FourierWrapper(wrapped_components::W, p_f::P,
            s::S
    ) where {N, S, P, W <: NTuple{N, AbstractOpticalComponent}}
        new{Trainable, S, W, P}(wrapped_components, p_f, s)
    end

    function FourierWrapper(p_f::FFTPlans,
            wrapped_components::Vararg{AbstractOpticalComponent})
        s = size(p_f.ft)
        @assert s == size(p_f.ift)
        d = fftdims(p_f.ft)
        @assert d == fftdims(p_f.ift)
        N = length(wrapped_components)
        @assert N > 0
        P = typeof(p_f)
        M = any(istrainable, wrapped_components) ? Trainable : Static
        S = Val{(s, d)}
        W = typeof(wrapped_components)
        new{M, S, W, P}(wrapped_components, p_f, S())
    end

    function FourierWrapper(u::ScalarField{U, Nd},
            wrapped_components::Vararg{AbstractOpticalComponent}
    ) where {Nd, U}
        u_plan = similar(u.electric)
        p_f = make_fft_plans(u_plan, Tuple(1:Nd))
        FourierWrapper(p_f, wrapped_components...)
    end
end

Functors.@functor FourierWrapper (wrapped_components,)

function get_data(p::FourierWrapper)
    if length(p.wrapped_components) > 1
        @warn "Calling get_data on a Fourier wrapper with multiple components \
        recovers only the data of the first component."
    end
    get_data(p.wrapped_components[1])
end

function get_all_data(p::FourierWrapper)
    (; wrapped_data = map(get_all_data, p.wrapped_components))
end

trainable(p::FourierWrapper{<:Trainable}) = (; wrapped_components = p.wrapped_components)

function propagate_wrapped_components(u::ScalarField, p::FourierWrapper,
        direction::Type{<:Direction})
    for c in p.wrapped_components
        u = propagate!(u, c, direction)
    end
    u
end

function propagate!(u::ScalarField, p::FourierWrapper, direction::Type{<:Direction})
    u = compute_ft!(p.p_f, u)
    u = propagate_wrapped_components(u, p, direction)
    u = compute_ift!(p.p_f, u)
    u
end

function propagate(u::ScalarField, p::FourierWrapper, direction::Type{<:Direction})
    propagate!(copy(u), p, direction)
end

function Base.merge(p1::FourierWrapper{M1, S}, p2::FourierWrapper{M2, S}) where {M1, M2, S}
    wrapped_components = (p1.wrapped_components..., p2.wrapped_components...)
    FourierWrapper(wrapped_components, p1.p_f, p1.s)
end

include("fourier_phase.jl")
include("fourier_mask.jl")
