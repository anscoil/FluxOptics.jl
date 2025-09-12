struct FourierWrapper{M, N, P} <: AbstractPureComponent{M}
    wrapped_components::NTuple{N, AbstractOpticalComponent}
    p_f::P

    function FourierWrapper(wrapped_components::NTuple{N, AbstractOpticalComponent},
            p_f::P) where {N, P}
        new{Trainable, N, P}(wrapped_components, p_f)
    end

    function FourierWrapper(u::ScalarField{U, Nd},
            wrapped_components::Vararg{AbstractOpticalComponent}
    ) where {Nd, U}
        u_plan = similar(u.data)
        p_f = make_fft_plans(u_plan, Tuple(1:Nd))
        N = length(wrapped_components)
        @assert N > 0
        P = typeof(p_f)
        M = any(istrainable, wrapped_components) ? Trainable : Static
        new{M, N, P}(wrapped_components, p_f)
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

trainable(p::FourierWrapper) = (; wrapped_components = p.wrapped_components)

function propagate_wrapped_components(u::ScalarField, p::FourierWrapper,
        direction::Type{Forward})
    for c in p.wrapped_components
        u = propagate!(u, c, Forward)
    end
    u
end

function propagate_wrapped_components(u::ScalarField, p::FourierWrapper,
        direction::Type{Backward})
    for c in reverse(p.wrapped_components)
        u = propagate!(u, c, Backward)
    end
    u
end

function propagate(u::ScalarField, p::FourierWrapper, direction::Type{<:Direction})
    u = compute_ft!(p.p_f, u)
    u = propagate_wrapped_components(u, p, direction)
    u = compute_ift!(p.p_f, u)
    u
end

include("fourier_phase.jl")
include("fourier_mask.jl")
