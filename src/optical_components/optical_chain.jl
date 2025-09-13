function (p::AbstractOpticalComponent)(u; forward::Bool = true, inplace::Bool = false)
    direction = forward ? Forward : Backward
    if inplace
        propagate!(u, p, direction)
    else
        propagate(u, p, direction)
    end
end

function (p::AbstractOpticalSource)()
    propagate(p)
end

struct OpticalChain{N, T <: NTuple{N, AbstractOpticalComponent}, K}
    layers::T
    kwargs::Ref{K}
end

Functors.@functor OpticalChain (layers,)

function OpticalChain(layers...)
    OpticalChain(layers, Ref((; forward = true, inplace = false)))
end

function Base.length(p::OpticalChain)
    length(p.layers)
end

function get_layers(p::OpticalChain)
    p.layers
end

function set_kwargs!(m::OpticalChain; kwargs...)
    try
        kwargs = NamedTuple([k == :direction ?
                             (v == Forward ? (:forward, true)
                              : (:forward, false)) : (k, v)
                             for (k, v) in pairs(kwargs)])
        m.kwargs[] = merge(m.kwargs[], kwargs)
    catch _
        error("Wrong keyword arguments. Choices are $(keys(m.kwargs[])).")
    end
end

function Base.:|>(a::AbstractOpticalComponent, b::AbstractOpticalComponent)
    if (isa(a, FourierWrapper) &&
        isa(b, FourierWrapper) &&
        a.s == b.s)
        OpticalChain(merge(a, b))
    else
        OpticalChain(a, b)
    end
end

function Base.:|>(a::AbstractOpticalComponent, b::OpticalChain)
    first_elem = b.layers[1]
    if (isa(a, FourierWrapper) &&
        isa(first_elem, FourierWrapper) &&
        a.s == first_elem.s)
        OpticalChain(merge(a, first_elem), b.layers[2, end]...)
    else
        OpticalChain(a, b.layers...)
    end
end

function Base.:|>(a::OpticalChain, b::AbstractOpticalComponent)
    last_elem = a.layers[end]
    if (isa(last_elem, FourierWrapper) &&
        isa(b, FourierWrapper) &&
        last_elem.s == b.s)
        OpticalChain(a.layers[1:(end - 1)]..., merge(last_elem, b))
    else
        OpticalChain(a.layers..., b)
    end
end

function Base.:|>(a::OpticalChain, b::OpticalChain)
    last_elem = a.layers[end]
    first_elem = b.layers[1]
    if (isa(last_elem, FourierWrapper) &&
        isa(first_elem, FourierWrapper) &&
        last_elem.s == first_elem.s)
        OpticalChain(a.layers[1:(end - 1)]...,
            merge(last_elem, first_elem),
            b.layers[2:end]...)
    else
        OpticalChain(a.layers..., b.layers...)
    end
end

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
    x = isnothing(x) ? m.layers[1]() : m.layers[1](x; kwargs...)
    d = IdDict{AbstractOpticalComponent, typeof(x)}()
    # After the first layer, `inplace=true` must be safe
    kwargs = merge(kwargs, (; inplace = true))
    iter_layers(m.layers[2:end], x, d; kwargs...)
end
