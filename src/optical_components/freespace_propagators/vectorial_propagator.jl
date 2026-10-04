struct VectorialPropagator{M, E, F} <: AbstractBidirectionalComponent{M}
    trainability::Val{M}
    z::Float64
    medium::E
    factors::F
    conjugate::Bool
end

Functors.@functor VectorialPropagator ()

function VectorialPropagator(medium::VectorialMediumModes, z::Real; conjugate::Bool = false)
    factors = PropagationFactors.(medium.modes, z, conjugate)
    VectorialPropagator(Val(Static), z, medium, factors, conjugate)
end

function VectorialPropagator(u::VectorialField, z::Real,
                             medium::Union{Permittivity, VectorialMediumModes} = 1.0;
                             conjugate::Bool = false)
    VectorialPropagator(VectorialMediumModes(u, medium), z; conjugate)
end

reference_medium(p::VectorialPropagator) = p.medium

function alloc_fp_state(u::VectorialField, p::VectorialPropagator)
    p.conjugate ? nothing : (; amp1 = similar(u.Ex), amp2 = similar(u.Ex))
end

function propagate!(u::VectorialField, state, ::Nothing, p::VectorialPropagator,
                    direction::Direction)
    launch_modal!(propagate_modes, (u.Ex, u.Ey, u.Hx, u.Hy), state_arrays(state),
                  (p.medium.modes, p.factors), direction)
    u
end

function propagate_adjoint!(u::VectorialField, ::Nothing, state, ::Nothing,
                            p::VectorialPropagator, direction::Direction)
    launch_modal!(propagate_modes_adjoint, (u.Ex, u.Ey, u.Hx, u.Hy), state_arrays(state),
                  (p.medium.modes, p.factors), direction)
    u
end
