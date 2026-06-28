module ChainRulesCoreExt

using ..GridUtils
using ..Metrics
using ..FFTutils
using ..Fields
using ..OpticalComponents
using ..OpticalComponents: get_preallocated_gradient, get_saved_buffer
using ..OpticalComponents: alloc_gradient, alloc_saved_buffer
using ..OpticalComponents: backpropagate!, backpropagate
using ..OpticalComponents: data_symbol, data_symbol_chain, auxiliary_trainable
using ..OpticalComponents: propagate_and_save, backpropagate_with_gradient
using ..OpticalComponents: propagate_and_save!, backpropagate_with_gradient!
using ..OpticalComponents: set_basis_projection!, apply_smoothing!, apply_projection!

using ChainRulesCore
using Functors: fleaves
using LinearAlgebra

ACTB = AbstractCustomComponent{Trainable{Buffered}}
ACTU = AbstractCustomComponent{Trainable{Unbuffered}}
ASTB = AbstractCustomSource{Trainable{Buffered}}
ASTU = AbstractCustomSource{Trainable{Unbuffered}}

function auxiliary_tangent(p, ∂c)
    inner = trainable(p)
    skip = data_symbol(p)
    NamedTuple(k => getproperty(∂c, k) for (k, _) in pairs(inner) if k !== skip)
end

function find_property(obj, sym::Symbol)
    hasproperty(obj, sym) && return getproperty(obj, sym)
    
    if obj isa ChainRulesCore.Tangent
        b = ChainRulesCore.backing(obj)
        if b isa Tuple
            for elem in b
                r = find_property(elem, sym)
                r !== nothing && return r
            end
        end
    end
    
    return nothing
end

function getproperty_nested(obj, symbols)
    result = getproperty(obj, first(symbols))
    foldl(Base.tail(symbols); init=result) do current, sym
        current === nothing && return nothing
        find_property(current, sym)
    end
end

function ChainRulesCore.rrule(::typeof(spatial_vectors),
                              ns::NTuple{Nd, Real},
                              ds::NTuple{Nd, T};
                              offset::NTuple{Nd, Real} = ntuple(_ -> 0, Nd)) where {Nd, T <: Real}
    result = spatial_vectors(ns, ds; offset)
    pullback(∂result) = NoTangent(), NoTangent(), NoTangent()
    return result, pullback
end

function ChainRulesCore.rrule(::typeof(propagate),
                              u,
                              p::AbstractCustomComponent{Static})
    v = propagate(u, p)
    function pullback(∂v)
        ∂u = backpropagate(∂v, p)
        return (NoTangent(), ∂u, NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(propagate),
                              u,
                              p::P) where {P <: ACTB}
    v = propagate_and_save(u, p)

    function pullback(∂v)
        ∂p = get_preallocated_gradient(p)
        u_saved = get_saved_buffer(p)
        ∂u, ∂p = backpropagate_with_gradient(∂v, u_saved, ∂p, p)
        return (NoTangent(), ∂u, Tangent{P}(; ∂p...))
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(propagate),
                              u,
                              p::P) where {P <: ACTU}
    u_saved = alloc_saved_buffer(u, p)
    v = propagate_and_save(u, u_saved, p)

    function pullback(∂v)
        ∂p = alloc_gradient(p)
        ∂u, ∂p = backpropagate_with_gradient(∂v, u_saved, ∂p, p)
        return (NoTangent(), ∂u, Tangent{P}(; ∂p...))
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(propagate!),
                              u,
                              p::AbstractCustomComponent{Static})
    v = propagate!(u, p)

    function pullback(∂v)
        ∂u = backpropagate!(∂v, p)
        return (NoTangent(), ∂u, NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(propagate!),
                              u,
                              p::P) where {P <: ACTB}
    v = propagate_and_save!(u, p)

    function pullback(∂v)
        ∂p = get_preallocated_gradient(p)
        u_saved = get_saved_buffer(p)
        ∂u, ∂p = backpropagate_with_gradient!(∂v, u_saved, ∂p, p)
        return (NoTangent(), ∂u, Tangent{P}(; ∂p...))
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(propagate!),
                              u,
                              p::P) where {P <: ACTU}
    u_saved = alloc_saved_buffer(u, p)
    v = propagate_and_save!(u, u_saved, p)

    function pullback(∂v)
        ∂p = alloc_gradient(p)
        ∂u, ∂p = backpropagate_with_gradient!(∂v, u_saved, ∂p, p)
        return (NoTangent(), ∂u, Tangent{P}(; ∂p...))
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(propagate), p::AbstractCustomSource{Static})
    v = propagate(p)

    function pullback(∂v)
        return (NoTangent(), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(propagate), p::P) where {P <: ASTB}
    v = propagate_and_save(p)

    function pullback(∂v)
        ∂p = get_preallocated_gradient(p)
        ∂p = backpropagate_with_gradient(∂v, ∂p, p)
        return (NoTangent(), Tangent{P}(; ∂p...))
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(propagate), p::P) where {P <: ASTU}
    v = propagate_and_save(p)

    function pullback(∂v)
        ∂p = alloc_gradient(p)
        ∂p = backpropagate_with_gradient(∂v, ∂p, p)
        return (NoTangent(), Tangent{P}(; ∂p...))
    end

    return v, pullback
end

function ChainRulesCore.rrule(::Type{<:ScalarField}, data::AbstractArray, ds, lambdas;
                              tilts=nothing)
    y = isnothing(tilts) ? ScalarField(data, ds, lambdas) : ScalarField(data, ds, lambdas; tilts)
    # y = ScalarField(data, ds, lambdas; tilts)
    function pullback(∂y)
        (NoTangent(), ∂y.electric, NoTangent(), NoTangent())
    end
    return y, pullback
end

function ChainRulesCore.rrule(::Type{<:ScalarField}, data::AbstractArray, ds,
                              lambdas::NamedTuple, tilts::NamedTuple)
    y = ScalarField(data, ds, lambdas, tilts)
    function pullback(∂y)
        (NoTangent(), ∂y.electric, NoTangent(), NoTangent(), NoTangent())
    end
    return y, pullback
end

function materialize(x::Base.ReshapedArray{T, N, <:Adjoint{T, <:AbstractArray}}) where {T, N}
    adj_materialized = copy(parent(x))
    reshape(adj_materialized, size(x))
end

materialize(x) = unthunk(x)

function ChainRulesCore.ProjectTo(u::ScalarField{U}) where {U}
    function (∂y)
        ∂y = unthunk(∂y)
        if ∂y.electric isa NoTangent
            NoTangent()
        else
            ScalarField(materialize(∂y.electric), u.ds, u.lambdas, u.tilts)
        end
    end
end

function ChainRulesCore.rrule(::typeof(compute_ft!), p_f, u)
    v = compute_ft!(p_f, u)

    function pullback(∂v)
        ∂u = compute_ift!(p_f, ∂v)
        return (NoTangent(), NoTangent(), ∂u)
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(compute_ift!), p_f, u)
    v = compute_ift!(p_f, u)

    function pullback(∂v)
        ∂u = compute_ft!(p_f, ∂v)
        return (NoTangent(), NoTangent(), ∂u)
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(pad), u::AbstractArray, ns::NTuple{Nd, Integer};
                              offset, pad_val) where {Nd}
    v = pad(u, ns; offset, pad_val)

    function pullback(∂v)
        ∂u = crop(∂v, size(u)[1:Nd]; offset)
        return (NoTangent(), ∂u, NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(crop), v::AbstractArray, ns::NTuple{Nd, Integer};
                              offset) where {Nd}
    u = crop(v, ns; offset)

    function pullback(∂u)
        ∂v = pad(∂u, size(v)[1:Nd]; offset, pad_val = 0)
        return (NoTangent(), ∂v, NoTangent())
    end

    return u, pullback
end

function compute_basis_projection!(proj_coeffs, r_basis, r_data)
    mul!(proj_coeffs, r_basis', r_data)
end

function ChainRulesCore.rrule(::typeof(set_basis_projection!),
                              p::P) where {P <: BasisProjectionWrapper}
    wrapped_component = set_basis_projection!(p)

    function pullback(∂c)
        ∂mapped_data = getproperty_nested(∂c, data_symbol_chain(wrapped_component))
        aux_data = auxiliary_tangent(wrapped_component, ∂c)
        if isbuffered(p)
            ∂p = p.∂p
            mul!(∂p.proj_coeffs, p.basis', reshape(∂mapped_data, :))
        else
            ∂p = (; proj_coeffs = p.basis' * reshape(∂mapped_data, :))
        end
        return (NoTangent(), Tangent{P}(; ∂p..., aux_data))
    end

    return wrapped_component, pullback
end

function ChainRulesCore.rrule(::typeof(apply_smoothing!),
                              p::P) where {P <: FourierSmoothingWrapper}
    wrapped_component = apply_smoothing!(p)

    function pullback(∂c)
        ∂mapped_data = getproperty_nested(∂c, data_symbol_chain(wrapped_component))
        aux_data = auxiliary_tangent(wrapped_component, ∂c)
        ∂p = isbuffered(p) ? p.∂p : (; buffer = similar(p.buffer))
        copyto!(∂p.buffer, ∂mapped_data)
        p.p_f.ft * ∂p.buffer
        ∂p.buffer .*= conj(p.filter)
        p.p_f.ift * ∂p.buffer
        
        return (NoTangent(), Tangent{P}(; ∂p..., aux_data))
    end

    return wrapped_component, pullback
end

function ChainRulesCore.rrule(::typeof(apply_projection!),
                              p::P) where {P <: DensityWrapper}
    wrapped_component = apply_projection!(p)
    
    function pullback(∂c)
        ∂mapped_data = getproperty_nested(∂c, data_symbol_chain(wrapped_component))
        aux_data = auxiliary_tangent(wrapped_component, ∂c)
        ∂p = isbuffered(p) ? p.∂p : (; D = similar(p.D), h = similar(p.h))
        σ_val = @. (p.mapped_data - p.offset) / p.h
        @. ∂p.D = ∂mapped_data * σ_val
        sum!(∂p.h, ∂p.D)
        @. ∂p.D *= p.h * p.β[] * (1 - σ_val)
        
        return (NoTangent(), Tangent{P}(; ∂p..., aux_data))
    end
    
    return wrapped_component, pullback
end

function ChainRulesCore.rrule(::typeof(compute_metric),
                              m::AbstractMetric,
                              u::NTuple{N, ScalarField}) where {N}
    c = compute_metric(m, u)

    function pullback(∂c)
        ∂c = map(c -> unthunk(c), unthunk(∂c))
        (NoTangent(), NoTangent(), backpropagate_metric(m, u, ∂c))
    end

    return c, pullback
end

include("bidirectional_chainrules.jl")

end
