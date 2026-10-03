struct VectorialFlatInterface{M, E1, E2, S, F} <: AbstractBidirectionalComponent{M}
    trainability::Val{M}
    ϵ1::E1
    ϵ2::E2
    modes_1::S
    modes_2::S
    fresnel::F
end

function VectorialFlatInterface(u::VectorialField, ϵ1::Permittivity, ϵ2::Permittivity)
    modes_1 = eigen_modes(u.Ex, u.ds, u.lambda, ϵ1)
    modes_2 = eigen_modes(u.Ex, u.ds, u.lambda, ϵ2)
    fresnel = compute_fresnel(modes_1, modes_2)
    VectorialFlatInterface(Val(Static), ϵ1, ϵ2, modes_1, modes_2, fresnel)
end

Functors.@functor VectorialFlatInterface ()

reference_medium_left(p::VectorialFlatInterface) = p.ϵ1

reference_medium_right(p::VectorialFlatInterface) = p.ϵ2

function alloc_fp_state(u::VectorialField, p::VectorialFlatInterface)
    (; amp1 = similar(u.Ex), amp2 = similar(u.Ex))
end

function interface_operands(p::VectorialFlatInterface, ::Forward)
    (p.modes_1, p.modes_2, p.fresnel.t12, p.fresnel.r21)
end

function interface_operands(p::VectorialFlatInterface, ::Backward)
    (p.modes_2, p.modes_1, p.fresnel.t21, p.fresnel.r12)
end

function cross_interface(m_in, m_out, t, r, ex, ey, hx, hy, s1, s2, direction::Direction)
    a_in = decompose(m_in, SVector(ex, ey, hx, hy), direction)
    a_st = SVector(s1, s2)
    a_out = t * a_in + r * a_st
    Ψ = recompose(m_out, a_out, direction) + recompose(m_out, a_st, reverse(direction))
    (Tuple(Ψ)..., Tuple(a_in)...)
end

function cross_interface_adjoint(m_in, m_out, t, r, ∂ex, ∂ey, ∂hx, ∂hy, ∂s1, ∂s2,
                                 direction::Direction)
    ∂Ψ_out = SVector(∂ex, ∂ey, ∂hx, ∂hy)
    ∂a_out = recompose_adjoint(m_out, ∂Ψ_out, direction)
    ∂a_st = r' * ∂a_out + recompose_adjoint(m_out, ∂Ψ_out, reverse(direction))
    ∂Ψ = decompose_adjoint(m_in, t' * ∂a_out + SVector(∂s1, ∂s2), direction)
    (Tuple(∂Ψ)..., Tuple(∂a_st)...)
end

function propagate!(u::VectorialField, state, ::Nothing, p::VectorialFlatInterface,
                    direction::Direction)
    m_in, m_out, t, r = interface_operands(p, direction)
    fields = (u.Ex, u.Ey, u.Hx, u.Hy, state.amp1, state.amp2)
    StructArray(fields) .= cross_interface.(m_in, m_out, t, r, fields..., direction)
    u
end

function propagate_adjoint!(u::VectorialField, ::Nothing, state, ::Nothing,
                            p::VectorialFlatInterface, direction::Direction)
    m_in, m_out, t, r = interface_operands(p, direction)
    fields = (u.Ex, u.Ey, u.Hx, u.Hy, state.amp1, state.amp2)
    StructArray(fields) .= cross_interface_adjoint.(m_in, m_out, t, r, fields..., direction)
    u
end
