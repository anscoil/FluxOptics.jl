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

Functors.@functor OpticalChain (layers,)

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

function compute_split_output(p::AbstractOpticalComponent, u; kwargs...)
    v = p(u; kwargs...)
    (v, nothing)
end

function compute_split_output(p::FieldProbe, u; kwargs...)
    p(u; kwargs...)
end

function iter_layers(layers, x, d; kwargs...)
    for layer in layers
        x, x_probe = compute_split_output(layer, x; kwargs...)
        if !isnothing(x_probe)
            d[layer] = x_probe
        end
    end
    (; out = x, probes = d)
end

function (m::OpticalChain{N})(x = nothing; call_kwargs...) where {N}
    kwargs = merge(m.kwargs[], call_kwargs)
    x = isnothing(x) ? m.layers[1](; call_kwargs...) : m.layers[1](x; kwargs...)
    d = IdDict{AbstractOpticalComponent, typeof(x)}()
    iter_layers(m.layers[2:end], x, d; kwargs...)
end

Flux.trainable(p::OpticalChain) = (; layers = p.layers)

Flux.trainable(p::AbstractOpticalComponent) = OpticalComponents.trainable(p)

Flux.@layer OpticalChain

Flux.@layer ASProp
Flux.@layer ASPropZ
Flux.@layer RSProp
Flux.@layer CollinsProp

Flux.@layer ScalarSource

Flux.@layer Phase
Flux.@layer ComplexMask
Flux.@layer TeaDOE

Flux.@layer FieldProbe

Flux.@layer GainSheet
