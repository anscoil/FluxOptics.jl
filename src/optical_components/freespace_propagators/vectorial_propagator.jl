struct VectorialPropagator{M, E, F, R} <: AbstractBidirectionalComponent{M}
    trainability::Val{M}
    z::Float64
    medium::E
    factors::F
    transports::R    # nothing: the opposite modes are kept in the state
end

Functors.@functor VectorialPropagator ()

"""
    VectorialPropagator(medium, z; stateful = true, keep_evanescent = false)

Free propagation over z. A stateful propagator keeps the modes travelling against each
pass in the fixed-point state, which is exact. A stateless one transports them across
z: exact for propagating modes; evanescent ones are cut unless keep_evanescent, since
their transport grows as exp(κz) (safe for thin layers only).
"""
function VectorialPropagator(medium::VectorialMediumModes, z::Real;
                             stateful::Bool = true, keep_evanescent::Bool = false)
    factors = propagation_factors.(medium.modes, z)
    transports = stateful ? nothing : transport_factors.(medium.modes, z; keep_evanescent)
    VectorialPropagator(Val(Static), z, medium, factors, transports)
end

function VectorialPropagator(u::VectorialField, z::Real,
                             medium::Union{Permittivity, VectorialMediumModes} = 1.0;
                             stateful::Bool = true, keep_evanescent::Bool = false)
    VectorialPropagator(VectorialMediumModes(u, medium), z; stateful, keep_evanescent)
end

reference_medium(p::VectorialPropagator) = p.medium

function alloc_fp_state(u::VectorialField, p::VectorialPropagator)
    isnothing(p.transports) ? (; amp1 = similar(u.Ex), amp2 = similar(u.Ex)) : nothing
end

function propagate!(u::VectorialField, state, ::Nothing, p::VectorialPropagator,
                    direction::Direction)
    launch_modal!(propagate_modes, (u.Ex, u.Ey, u.Hx, u.Hy), state_arrays(state),
                  modal_coefs(p.medium, p.factors, p.transports), direction)
    u
end

function propagate_adjoint!(u::VectorialField, ::Nothing, state, ::Nothing,
                            p::VectorialPropagator, direction::Direction)
    launch_modal!(propagate_modes_adjoint, (u.Ex, u.Ey, u.Hx, u.Hy), state_arrays(state),
                  modal_coefs(p.medium, p.factors, p.transports), direction)
    u
end
