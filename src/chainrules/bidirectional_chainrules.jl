using ..OpticalComponents: apply_implicit, combine_implicit
using ..OpticalComponents: fp_solve_adjoint!, compute_roundtrip_adjoint!
using ..OpticalComponents: alloc_activations, alloc_gradient

copy_tangent!(dest::AbstractArray, src::AbstractArray) = copyto!(dest, src)
copy_tangent!(dest::AbstractArray, src) = fill!(dest, 0)

tangent_component(∂u, name::Symbol) = getproperty(∂u, name)
tangent_component(::AbstractZero, ::Symbol) = ZeroTangent()

function set_adjoint_source!(p::ScalarWaveSource, ∂u)
    copy_tangent!(p.u0.E, tangent_component(∂u, :E))
    copy_tangent!(p.u0.dzE, tangent_component(∂u, :dzE))
    p.u0
end

function set_adjoint_source!(p::VectorialSource, ∂u)
    copy_tangent!(p.u0.Ex, tangent_component(∂u, :Ex))
    copy_tangent!(p.u0.Ey, tangent_component(∂u, :Ey))
    copy_tangent!(p.u0.Hx, tangent_component(∂u, :Hx))
    copy_tangent!(p.u0.Hy, tangent_component(∂u, :Hy))
    p.u0
end

function ChainRulesCore.rrule(::typeof(apply_implicit), ufr, s, solver; kwargs...)
    function pullback(∂u_out)
        ∂uf, ∂ur = ∂u_out
        s_in, s_out = s.s_in_adj, s.s_out_adj
        set_adjoint_source!(s_in, ∂ur)
        set_adjoint_source!(s_out, ∂uf)
        fp_state_adj = fp_solve_adjoint!(s, solver; kwargs...)
        copyto!(s.tmp_state, fp_state_adj)
        ∂ufr = compute_roundtrip_adjoint!(s, s_in, s_out, s.tmp_state)
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

function ChainRulesCore.rrule(::typeof(propagate!), u, state, activations,
                              p::P, direction::Direction
                              ) where {P <: AbstractBidirectionalComponent{Trainable}}
    activations = isnothing(activations) ? alloc_activations(u, p, direction) : activations
    v = propagate!(u, state, activations, p, direction)

    function pullback(∂v)
        ∂p = alloc_gradient(p)
        ∂u = propagate_adjoint!(∂v, ∂p, state, activations, p, direction)
        return (NoTangent(), ∂u, NoTangent(), NoTangent(), Tangent{P}(; ∂p...), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(propagate!), u, state, ::Nothing,
                              p::P, direction::Direction
                              ) where {P <: AbstractBidirectionalComponent{Static}}
    v = propagate!(u, state, p, direction)

    function pullback(∂v)
        ∂u = propagate_adjoint!(∂v, state, p, direction)
        return (NoTangent(), ∂u, NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(propagate!), u, p::P, direction::Direction
                              ) where {P <: AbstractBidirectionalSource}
    v = propagate!(u, p, direction)

    function pullback(∂v)
        ∂u = propagate_adjoint!(∂v, p, direction)
        return (NoTangent(), ∂u, NoTangent(), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(emit), p::P
                              ) where {P <: AbstractBidirectionalSource}
    u = emit(p)
    pullback(∂u) = NoTangent(), NoTangent()
    return u, pullback
end
