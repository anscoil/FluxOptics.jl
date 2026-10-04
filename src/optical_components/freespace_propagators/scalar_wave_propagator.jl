struct ScalarWavePropagator{M, E, F} <: AbstractBidirectionalComponent{M}
    trainability::Val{M}
    z::Float64
    medium::E
    factors::F
    conjugate::Bool
end

Functors.@functor ScalarWavePropagator ()

function ScalarWavePropagator(medium::ScalarMediumModes, z::Real; conjugate::Bool = false)
    factors = PropagationFactors.(medium.modes, z, conjugate)
    ScalarWavePropagator(Val(Static), z, medium, factors, conjugate)
end

function ScalarWavePropagator(u::ScalarWaveField, z::Real,
                              medium::Union{Number, ScalarMediumModes} = 1.0;
                              conjugate::Bool = false)
    ScalarWavePropagator(ScalarMediumModes(u, medium), z; conjugate)
end

reference_medium(p::ScalarWavePropagator) = p.medium

function alloc_fp_state(u::ScalarWaveField, p::ScalarWavePropagator)
    p.conjugate ? nothing : (; amp = similar(u.E))
end

function propagate!(u::ScalarWaveField, state, ::Nothing,
                    p::ScalarWavePropagator, direction::Direction)
    launch_modal!(propagate_modes, (u.E, u.dzE), state_arrays(state),
                  (p.medium.modes, p.factors), direction)
    u
end

function propagate_adjoint!(u::ScalarWaveField, ::Nothing, state, ::Nothing,
                            p::ScalarWavePropagator, direction::Direction)
    launch_modal!(propagate_modes_adjoint, (u.E, u.dzE), state_arrays(state),
                  (p.medium.modes, p.factors), direction)
    u
end
