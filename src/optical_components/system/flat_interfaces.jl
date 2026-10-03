include("scalar_flat_interface.jl")
include("vectorial_flat_interface.jl")

struct NoInterface{M} <: AbstractBidirectionalComponent{M}
    trainability::Val{M}
end

NoInterface() = NoInterface(Val(Static))

alloc_fp_state(u::AbstractField, p::NoInterface) = nothing

propagate!(u::AbstractField, state, ::Nothing, p::NoInterface, ::Direction) = u

propagate_adjoint!(u::AbstractField, state, ::Nothing, p::NoInterface, ::Direction) = u

function FlatInterface(u::AbstractField, ::Nothing, ::Nothing)
    NoInterface()
end

function FlatInterface(u::ScalarWaveField, n1::Number, n2::Number)
    ScalarFlatInterface(u, n1, n2)
end

function FlatInterface(u::ScalarWaveField, n1::Number, n2::Nothing)
    ScalarFlatInterface(u, n1, n1)
end

function FlatInterface(u::ScalarWaveField, n1::Nothing, n2::Number)
    ScalarFlatInterface(u, n2, n2)
end

function FlatInterface(u::VectorialField, ϵ1::Permittivity, ϵ2::Permittivity)
    VectorialInterface(u, ϵ1, ϵ2)
end

function FlatInterface(u::VectorialField, ϵ1::Permittivity, ϵ2::Nothing)
    VectorialInterface(u, ϵ1, ϵ1)
end

function FlatInterface(u::VectorialField, ϵ1::Nothing, ϵ2::Permittivity)
    VectorialInterface(u, ϵ2, ϵ2)
end
