using Flux
using Functors
using AbstractFFTs

function (p::AbstractOpticalComponent)(u; direction::Type{<:Direction} = Forward,
        inplace::Val{B} = Val(false)) where {B}
    if B
        propagate!(u, p, direction)
    else
        propagate(u, p, direction)
    end
end

function (p::AbstractOpticalSource)(; direction::Type{<:Direction} = Forward)
    propagate(p, direction)
end

struct OpticalChain{N, T <: NTuple{N, AbstractOpticalComponent}, K}
    layers::T
    kwargs::K
end

function OpticalChain(layers...; kwargs...)
    OpticalChain(layers, kwargs)
end

function (m::OpticalChain)(x; call_kwargs...)
    kwargs = merge(m.kwargs, call_kwargs)
    for layer in m.layers
        x = layer(x; kwargs...)
    end
    return x
end

function (m::OpticalChain)(; call_kwargs...)
    kwargs = merge(m.kwargs, call_kwargs)
    x = m.layers[1](; call_kwargs...)
    for layer in m.layers[2:end]
        x = layer(x; kwargs...)
    end
    return x
end

Flux.trainable(p::OpticalChain) = (; layers = p.layers)

Flux.trainable(p::AbstractOpticalComponent) = OpticalComponents.trainable(p)

Flux.@layer OpticalChain

Functors.@leaf AbstractFFTs.Plan

Flux.@layer ASProp{Static}
Flux.@layer RSProp{Static}

Flux.@layer Phase
Flux.@layer Seeder
