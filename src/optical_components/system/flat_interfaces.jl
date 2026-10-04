struct FlatInterface{M, E1, E2, F} <: AbstractBidirectionalComponent{M}
    trainability::Val{M}
    m1::E1
    m2::E2
    fresnel::F
end

Functors.@functor FlatInterface ()

function FlatInterface(m1::MediumModes{B}, m2::MediumModes{B}) where {B}
    FlatInterface(Val(Static), m1, m2, compute_fresnel(m1, m2))
end

FlatInterface(m1::MediumModes, ::Nothing) = FlatInterface(m1, m1)
FlatInterface(::Nothing, m2::MediumModes) = FlatInterface(m2, m2)

reference_medium_left(p::FlatInterface) = p.m1
reference_medium_right(p::FlatInterface) = p.m2

interface_media(p::FlatInterface, ::Forward) = (p.m1.modes, p.m2.modes)
interface_media(p::FlatInterface, ::Backward) = (p.m2.modes, p.m1.modes)

alloc_fp_state(u::ScalarWaveField, p::FlatInterface) = (; amp = similar(u.E))
alloc_fp_state(u::VectorialField, p::FlatInterface) = (; amp1 = similar(u.Ex), amp2 = similar(u.Ex))

function cross_interface(m_in, m_out, f::FresnelCoefficients, direction::Direction,
                         ψ::Tuple, stored::Tuple)
    a_in = decompose(m_in, SVector(ψ), direction)
    a_st = SVector(stored)
    a_out = transmission(f, direction) * a_in + reflection(f, direction) * a_st
    Ψ = recompose(m_out, a_out, direction) + recompose(m_out, a_st, reverse(direction))
    Tuple(Ψ), Tuple(a_in)
end

function cross_interface_adjoint(m_in, m_out, f::FresnelCoefficients, direction::Direction,
                                 ∂ψ::Tuple, ∂stored::Tuple)
    ∂Ψ_out = SVector(∂ψ)
    ∂a_out = recompose_adjoint(m_out, ∂Ψ_out, direction)
    ∂a_st = reflection(f, direction)' * ∂a_out + recompose_adjoint(m_out, ∂Ψ_out, reverse(direction))
    ∂Ψ = decompose_adjoint(m_in, transmission(f, direction)' * ∂a_out + SVector(∂stored), direction)
    Tuple(∂Ψ), Tuple(∂a_st)
end

# function propagate!(u::ScalarWaveField, state, ::Nothing, p::FlatInterface,
#                     direction::Direction)
#     m_in, m_out = interface_media(p, direction)
#     ψ = StructArray((u.E, u.dzE))
#     a_st = StructArray((state.amp,))
#     StructArray((ψ, a_st)) .= cross_interface.(m_in, m_out, p.fresnel, direction, ψ, a_st)
#     u
# end

# function propagate_adjoint!(u::ScalarWaveField, ::Nothing, state, ::Nothing,
#                             p::FlatInterface, direction::Direction)
#     m_in, m_out = interface_media(p, direction)
#     ψ = StructArray((u.E, u.dzE))
#     a_st = StructArray((state.amp,))
#     StructArray((ψ, a_st)) .= cross_interface_adjoint.(m_in, m_out, p.fresnel, direction, ψ, a_st)
#     u
# end

function propagate!(u::ScalarWaveField, state, ::Nothing,
                    p::FlatInterface, direction::Direction)
    m_in, m_out = interface_media(p, direction)
    launch_modal!(cross_interface, (u.E, u.dzE), (state.amp,),
                  (m_in, m_out, p.fresnel), direction)
    u
end

function propagate_adjoint!(u::ScalarWaveField, ::Nothing, state, ::Nothing,
                            p::FlatInterface, direction::Direction)
    m_in, m_out = interface_media(p, direction)
    launch_modal!(cross_interface_adjoint, (u.E, u.dzE), (state.amp,),
                  (m_in, m_out, p.fresnel), direction)
    u
end

# function propagate!(u::VectorialField, state, ::Nothing, p::FlatInterface,
#                     direction::Direction)
#     m_in, m_out = interface_media(p, direction)
#     ψ = StructArray((u.Ex, u.Ey, u.Hx, u.Hy))
#     a_st = StructArray((state.amp1, state.amp2))
#     StructArray((ψ, a_st)) .= cross_interface.(m_in, m_out, p.fresnel, direction, ψ, a_st)
#     u
# end

# function propagate_adjoint!(u::VectorialField, ::Nothing, state, ::Nothing,
#                             p::FlatInterface, direction::Direction)
#     m_in, m_out = interface_media(p, direction)
#     ψ = StructArray((u.Ex, u.Ey, u.Hx, u.Hy))
#     a_st = StructArray((state.amp1, state.amp2))
#     StructArray((ψ, a_st)) .= cross_interface_adjoint.(m_in, m_out, p.fresnel, direction, ψ, a_st)
#     u
# end

function propagate!(u::VectorialField, state, ::Nothing,
                    p::FlatInterface, direction::Direction)
    m_in, m_out = interface_media(p, direction)
    launch_modal!(cross_interface, (u.Ex, u.Ey, u.Hx, u.Hy),
                  (state.amp1, state.amp2), (m_in, m_out, p.fresnel), direction)
    u
end

function propagate_adjoint!(u::VectorialField, ::Nothing, state, ::Nothing,
                            p::FlatInterface, direction::Direction)
    m_in, m_out = interface_media(p, direction)
    launch_modal!(cross_interface_adjoint, (u.Ex, u.Ey, u.Hx, u.Hy),
                  (state.amp1, state.amp2), (m_in, m_out, p.fresnel), direction)
    u
end

struct NoInterface{M} <: AbstractBidirectionalComponent{M}
    trainability::Val{M}
end

NoInterface() = NoInterface(Val(Static))

alloc_fp_state(u::AbstractField, p::NoInterface) = nothing

propagate!(u::AbstractField, state, ::Nothing, p::NoInterface, ::Direction) = u

propagate_adjoint!(u::AbstractField, ::Nothing, state, ::Nothing,
                   p::NoInterface, ::Direction) = u

FlatInterface(::Nothing, ::Nothing) = NoInterface()
