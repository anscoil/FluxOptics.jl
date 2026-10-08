struct VectorialSource{U, M} <: AbstractBidirectionalSource{U}
    u0::VectorialField{U}
    uf::VectorialField{U}
    medium::M
end

Functors.@functor VectorialSource ()

function VectorialSource(u::VectorialField,
                         medium::Union{Permittivity, VectorialMediumModes} = 1.0)
    VectorialSource(copy(u), similar(u), VectorialMediumModes(u, medium))
end

Base.size(p::VectorialSource) = size(p.u0)
Base.size(p::VectorialSource, k::Integer) = size(p.u0, k)

reference_medium(p::VectorialSource) = p.medium

# function propagate!(u::VectorialField, p::VectorialSource, direction::Direction)
#     fields = (u.Ex, u.Ey, u.Hx, u.Hy)
#     src = (p.u0.Ex, p.u0.Ey, p.u0.Hx, p.u0.Hy)
#     StructArray(fields) .= inject_source.(fields, src, p.medium.modes, direction)
#     u
# end

# function propagate_adjoint!(u::VectorialField, p::VectorialSource, direction::Direction)
#     fields = (u.Ex, u.Ey, u.Hx, u.Hy)
#     StructArray(fields) .= inject_source_adjoint.(fields, (), p.medium.modes, direction)
#     u
# end

function propagate!(u::VectorialField, p::VectorialSource, direction::Direction)
    launch_combine!(inject_source, (u.Ex, u.Ey, u.Hx, u.Hy),
                    (p.u0.Ex, p.u0.Ey, p.u0.Hx, p.u0.Hy), (p.medium.modes,), direction)
    u
end

function propagate_adjoint!(u::VectorialField, p::VectorialSource, direction::Direction)
    launch_combine!(inject_source_adjoint, (u.Ex, u.Ey, u.Hx, u.Hy), (),
                    (p.medium.modes,), direction)
    u
end

emit(p::VectorialSource) = copyto!(p.uf, p.u0)

function Base.zero(p::VectorialSource;
                   medium::Union{Permittivity, VectorialMediumModes} = p.medium)
    VectorialSource(zero(p.u0), similar(p.u0), VectorialMediumModes(p.u0, medium))
end

Base.copy(p::VectorialSource) = VectorialSource(copy(p.u0), similar(p.u0), p.medium)
Base.fill!(p::VectorialSource, u0::VectorialField) = copyto!(p.u0, u0)

get_source(p::VectorialSource) = p.u0
