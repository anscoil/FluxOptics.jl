using ..OpticalComponents: apply_implicit, combine_implicit, apply_spectral_projection!
using ..OpticalComponents: fp_solve_adjoint!, compute_roundtrip_adjoint!
using ..OpticalComponents: alloc_activations, alloc_gradient

function Base.:+(a::NamedTuple{(:electric, :electric_dz, :ds, :lambdas)}, b::ScalarWaveField)
    electric = isnothing(a.electric) ? b.electric : a.electric + b.electric
    electric_dz = isnothing(a.electric_dz) ? b.electric_dz : a.electric_dz + b.electric_dz
    ScalarWaveField(electric, electric_dz, b.ds, b.lambdas)
end

Base.:+(b::ScalarWaveField, a::NamedTuple{(:electric, :electric_dz, :ds, :lambdas)}) = a + b

function set_adjoint_source!(p::ScalarWaveSource, ∂u)
    if ∂u isa Union{ZeroTangent, NoTangent}
        fill!(p.u0.electric, 0)
        fill!(p.u0.electric_dz, 0)
        return p.u0
    end
    if ∂u.electric isa AbstractArray
        copyto!(p.u0.electric, ∂u.electric)
    else
        fill!(p.u0.electric, 0)
    end
    if ∂u.electric_dz isa AbstractArray
        copyto!(p.u0.electric_dz, ∂u.electric_dz)
    else
        fill!(p.u0.electric_dz, 0)
    end
    p.u0
end

function ChainRulesCore.rrule(::typeof(apply_implicit), ufr, s, solver;
                              spectral_projection = false, kwargs...)
    function pullback(∂u_out)
        ∂uf, ∂ur = ∂u_out
        s_in, s_out = s.s_in_adj, s.s_out_adj
        set_adjoint_source!(s_in, ∂ur)
        set_adjoint_source!(s_out, ∂uf)
        fp_state_adj = fp_solve_adjoint!(s, solver; spectral_projection, kwargs...)
        copyto!(s.tmp_state, fp_state_adj)
        ∂ufr = compute_roundtrip_adjoint!(s, s_in, s_out, s.tmp_state; spectral_projection)
        return NoTangent(), ∂ufr, NoTangent(), NoTangent()
    end
    return ufr, pullback
end

function ChainRulesCore.rrule(::typeof(combine_implicit), ufr, ufri)
    function pullback(∂ufr)
        return NoTangent(), ∂ufr, ∂ufr
    end
    return ufr, pullback
end

function ChainRulesCore.rrule(::typeof(apply_spectral_projection!), s, fp_state)
    state = apply_spectral_projection!(s, fp_state)
    function pullback(∂state)
        apply_spectral_projection!(s, ∂state)
        return NoTangent(), NoTangent(), ∂state
    end
    return state, pullback
end

function ChainRulesCore.rrule(::typeof(propagate!), u, state, p::P
                              ) where {P <: AbstractBidirectionalComponent{Trainable}}
    activations = alloc_activations(u, p)
    v = propagate!(u, state, activations, p)

    function pullback(∂v)
        ∂p = alloc_gradient(p)
        ∂u = propagate_adjoint!(∂v, ∂p, state, activations, p)
        return (NoTangent(), ∂u, NoTangent(), Tangent{P}(; ∂p...))
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(propagate!), u, state, p::P
                              ) where {P <: AbstractBidirectionalComponent{Static}}
    v = propagate!(u, state, p)

    function pullback(∂v)
        ∂u = propagate_adjoint!(∂v, state, p)
        return (NoTangent(), ∂u, NoTangent(), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(inverse_propagate!), u, state, p::P
                              ) where {P <: AbstractBidirectionalComponent{Trainable}}
    activations = alloc_activations(u, p)
    v = inverse_propagate!(u, state, activations, p)

    function pullback(∂v)
        ∂p = alloc_gradient(p)
        ∂u = inverse_propagate_adjoint!(∂v, ∂p, state, activations, p)
        return (NoTangent(), ∂u, NoTangent(), Tangent{P}(; ∂p...))
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(inverse_propagate!), u, state, p::P
                              ) where {P <: AbstractBidirectionalComponent{Static}}
    v = inverse_propagate!(u, state, p)

    function pullback(∂v)
        ∂u = inverse_propagate_adjoint!(∂v, state, p)
        return (NoTangent(), ∂u, NoTangent(), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(propagate!), u, p::P
                              ) where {P <: AbstractBidirectionalSource}
    v = propagate!(u, p)

    function pullback(∂v)
        ∂u = propagate_adjoint!(∂v, p)
        return (NoTangent(), ∂u, NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(inverse_propagate!), u, p::P
                              ) where {P <: AbstractBidirectionalSource}
    v = inverse_propagate!(u, p)

    function pullback(∂v)
        ∂u = inverse_propagate_adjoint!(∂v, p)
        return (NoTangent(), ∂u, NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(propagate), p::P
                              ) where {P <: AbstractBidirectionalSource}
    u = propagate(p)
    pullback(∂u) = NoTangent(), NoTangent()
    return u, pullback
end
