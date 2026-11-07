struct ScalarWavePropagator{M, E, F, R} <: AbstractBidirectionalComponent{M}
    trainability::Val{M}
    z::Float64
    medium::E
    factors::F
    transports::R    # nothing: the opposite modes are kept in the state
end

Functors.@functor ScalarWavePropagator ()

"""
    ScalarWavePropagator(medium, z; stateful = true, keep_evanescent = false)

Free propagation over z. A stateful propagator keeps the modes travelling against each
pass in the fixed-point state, which is exact. A stateless one transports them across
z: exact for propagating modes; evanescent ones are cut unless keep_evanescent, since
their transport grows as exp(κz) (safe for thin layers only).
"""
function ScalarWavePropagator(medium::ScalarMediumModes, z::Real;
                              stateful::Bool = true, keep_evanescent::Bool = false)
    factors = propagation_factors.(medium.modes, z)
    transports = stateful ? nothing : transport_factors.(medium.modes, z; keep_evanescent)
    ScalarWavePropagator(Val(Static), z, medium, factors, transports)
end

function ScalarWavePropagator(u::ScalarWaveField, z::Real,
                              medium::Union{Number, ScalarMediumModes} = 1.0;
                              stateful::Bool = true, keep_evanescent::Bool = false)
    ScalarWavePropagator(ScalarMediumModes(u, medium), z; stateful, keep_evanescent)
end

reference_medium(p::ScalarWavePropagator) = p.medium

function alloc_fp_state(u::ScalarWaveField, p::ScalarWavePropagator)
    isnothing(p.transports) ? (; amp = similar(u.E)) : nothing
end

function propagate!(u::ScalarWaveField, state, ::Nothing,
                    p::ScalarWavePropagator, direction::Direction)
    launch_modal!(propagate_modes, (u.E, u.dzE), state_arrays(state),
                  modal_coefs(p.medium, p.factors, p.transports), direction)
    u
end

function propagate_adjoint!(u::ScalarWaveField, ::Nothing, state, ::Nothing,
                            p::ScalarWavePropagator, direction::Direction)
    launch_modal!(propagate_modes_adjoint, (u.E, u.dzE), state_arrays(state),
                  modal_coefs(p.medium, p.factors, p.transports), direction)
    u
end
