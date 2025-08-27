using Flux
using Functors
using AbstractFFTs
using LRUCache

function (p::AbstractOpticalComponent)(u; direction::Type{<:Direction} = Forward,
        inplace::Bool = false)
    if inplace
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
    kwargs::Ref{K}
end

function OpticalChain(layers...)
    OpticalChain(layers, Ref((; inplace = false)))
end

function set_kwargs!(m::OpticalChain; kwargs...)
    try
        m.kwargs[] = merge(m.kwargs[], kwargs)
    catch _
        error("Wrong keyword arguments. Choices are $(keys(m.kwargs[])).")
    end
end

Base.:|>(a::AbstractOpticalComponent, b::AbstractOpticalComponent) = OpticalChain(a, b)

Base.:|>(a::AbstractOpticalComponent, b::OpticalChain) = OpticalChain(a, b.layers...)

Base.:|>(a::OpticalChain, b::AbstractOpticalComponent) = OpticalChain(a.layers..., b)

Base.:|>(a::OpticalChain, b::OpticalChain) = OpticalChain(a.layers..., b.layers...)

function Base.:|>(m::OpticalChain, kwargs::NamedTuple)
    set_kwargs!(m; kwargs...)
    m
end

function Base.:|>(p::AbstractOpticalComponent, kwargs::NamedTuple)
    m = OpticalChain(p)
    set_kwargs!(m; kwargs...)
    m
end

function (m::OpticalChain)(x; call_kwargs...)
    kwargs = merge(m.kwargs[], call_kwargs)
    for layer in m.layers
        x = layer(x; kwargs...)
    end
    return x
end

function (m::OpticalChain)(; call_kwargs...)
    kwargs = merge(m.kwargs[], call_kwargs)
    x = m.layers[1](; call_kwargs...)
    for layer in m.layers[2:end]
        x = layer(x; kwargs...)
    end
    return x
end

Flux.trainable(p::OpticalChain) = (; layers = p.layers)

Flux.trainable(p::AbstractOpticalComponent) = OpticalComponents.trainable(p)

Functors.@functor OpticalChain (layers,)
Flux.@layer OpticalChain

Functors.@functor ASProp ()
Functors.@functor ASPropZ (z,)
Functors.@functor RSProp ()
Functors.@functor CollinsProp ()
Flux.@layer ASProp
Flux.@layer ASPropZ
Flux.@layer RSProp
Flux.@layer CollinsProp

Functors.@functor ScalarSource (u0,)
Flux.@layer ScalarSource

Functors.@functor Phase (ϕ,)
Flux.@layer Phase

Functors.@functor TeaDOE (h,)
Flux.@layer TeaDOE
