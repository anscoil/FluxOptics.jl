function inject_source(m::AbstractModeBasis, direction::Direction, ψ::Tuple, ψ_src::Tuple)
    Tuple(project(m, SVector(ψ), direction) +
        project(m, SVector(ψ_src), reverse(direction)))
end

function inject_source_adjoint(m::AbstractModeBasis, direction::Direction,
                               ∂ψ::Tuple, ::Tuple{})
    Tuple(project_adjoint(m, SVector(∂ψ), direction))
end

@kernel function source_kernel!(f_inject, fields, src, modes, direction)
    I = @index(Global, Cartesian)
    m = modes[I]
    for J in CartesianIndices(axes(first(fields))[3:end])
        ψ = map(a -> a[I, J], fields)
        ψ_src = map(a -> a[I, J], src)
        out = f_inject(m, direction, ψ, ψ_src)
        map((a, v) -> a[I, J] = v, fields, out)
    end
end

function launch_source!(f_inject, fields::Tuple, src::Tuple, modes, direction::Direction)
    source_kernel!(get_backend(first(fields)))(
        f_inject, fields, src, modes, direction;
        ndrange = size(first(fields))[1:2])
end

include("scalar_wave_source.jl")
include("vectorial_source.jl")
