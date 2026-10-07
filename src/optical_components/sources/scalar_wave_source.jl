struct ScalarWaveSource{U, M} <: AbstractBidirectionalSource{U}
    u0::ScalarWaveField{U}
    uf::ScalarWaveField{U}
    medium::M
end

Functors.@functor ScalarWaveSource ()

function ScalarWaveSource(u::ScalarWaveField,
                          medium::Union{Number, ScalarMediumModes} = 1.0)
    ScalarWaveSource(copy(u), similar(u), ScalarMediumModes(u, medium))
end

Base.size(p::ScalarWaveSource) = size(p.u0)
Base.size(p::ScalarWaveSource, k::Integer) = size(p.u0, k)

reference_medium(p::ScalarWaveSource) = p.medium

function propagate!(u::ScalarWaveField, p::ScalarWaveSource, direction::Direction)
    launch_combine!(inject_source, (u.E, u.dzE), (p.u0.E, p.u0.dzE), (p.medium.modes,),
                    direction)
    u
end

function propagate_adjoint!(u::ScalarWaveField, p::ScalarWaveSource, direction::Direction)
    launch_combine!(inject_source_adjoint, (u.E, u.dzE), (), (p.medium.modes,), direction)
    u
end

emit(p::ScalarWaveSource) = copyto!(p.uf, p.u0)

function Base.zero(p::ScalarWaveSource;
                   medium::Union{Number, ScalarMediumModes} = p.medium)
    ScalarWaveSource(zero(p.u0), similar(p.u0), ScalarMediumModes(p.u0, medium))
end

Base.copy(p::ScalarWaveSource) = ScalarWaveSource(copy(p.u0), similar(p.u0), p.medium)
Base.fill!(p::ScalarWaveSource, u0::ScalarWaveField) = copyto!(p.u0, u0)

get_source(p::ScalarWaveSource) = p.u0
