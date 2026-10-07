function inject_source(m::AbstractModeBasis, direction::Direction, ψ::Tuple, ψ_src::Tuple)
    Tuple(project(m, SVector(ψ), direction) +
        project(m, SVector(ψ_src), reverse(direction)))
end

function inject_source_adjoint(m::AbstractModeBasis, direction::Direction,
                               ∂ψ::Tuple, ::Tuple{})
    Tuple(project_adjoint(m, SVector(∂ψ), direction))
end

@kernel function combine_kernel!(f, fields, inputs, coefs, direction)
    I = @index(Global, Cartesian)
    c = map(a -> a[I], coefs)
    for J in CartesianIndices(axes(first(fields))[3:end])
        ψ = map(a -> a[I, J], fields)
        ψ_in = map(a -> a[I, J], inputs)
        out = f(c..., direction, ψ, ψ_in)
        map((a, v) -> a[I, J] = v, fields, out)
    end
end

function launch_combine!(f, fields::Tuple, inputs::Tuple, coefs::Tuple,
                         direction::Direction)
    combine_kernel!(get_backend(first(fields)))(
        f, fields, inputs, coefs, direction; ndrange = size(first(fields))[1:2])
end

include("scalar_wave_source.jl")
include("vectorial_source.jl")
