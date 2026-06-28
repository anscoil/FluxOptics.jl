using ..OpticalComponents: apply_implicit, combine_implicit, apply_spectral_projection!

function ChainRulesCore.rrule(::typeof(apply_implicit), uf, ur, s, solver;
                              spectral_projection = false, kwargs...)
    function pullback(∂u_out)
        ∂uf, ∂ur = ∂u_out
        s_in, s_out = s.s_in_adj, s.s_out_adj
        fill!(s_in, ∂ur)
        fill!(s_out, ∂uf)
        fp_state_adj = fp_solve_adjoint!(s, solver; spectral_projection, kwargs...)
        copyto!(s.tmp_state, fp_state_adj)
        ∂uf, ∂ur = compute_roundtrip_adjoint!(s, s_in, s_out, s.tmp_state;
                                              spectral_projection)
        return NoTangent(), ∂uf, ∂ur, NoTangent(), NoTangent()
    end
    return (uf, ur), pullback
end

function ChainRulesCore.rrule(::typeof(combine_implicit), uf, ur, ufi, uri)
    function pullback(∂u_out)
        ∂uf, ∂ur = ∂u_out
        return NoTangent(), ∂uf, ∂ur, ∂uf, ∂ur
    end
    return (uf, ur), pullback
end

function ChainRulesCore.rrule(::typeof(apply_spectral_projection!), s, fp_state)
    state = apply_spectral_projection!(s, fp_state)
    function pullback(∂state)
        apply_spectral_projection!(s, ∂state)
        return NoTangent(), NoTangent(), ∂state
    end
    return state, pullback
end
