struct PropagationFactors{V}
    fwd::Tuple{V, V}  # forward pass: (forward modes, backward modes)
    bwd::Tuple{V, V}  # backward pass: (backward modes, forward modes)
end

propagation_factors(f::PropagationFactors, ::Forward) = f.fwd
propagation_factors(f::PropagationFactors, ::Backward) = f.bwd

function PropagationFactors(m::AbstractModeBasis, z::Real, conjugate::Bool)
    kz = eigenvalues(m)
    iz = im * real(eltype(kz))(z)
    f, b = mode_indices(m, Forward()), mode_indices(m, Backward())
    exp_p = exp.(iz .* kz)
    exp_m = exp.(-iz .* kz)
    fwd = (exp_p[f], conjugate ? conj.(exp_m[b]) : exp_p[b])
    bwd = (exp_m[b], conjugate ? conj.(exp_p[f]) : exp_m[f])
    PropagationFactors(fwd, bwd)
end

function propagate_modes(m::AbstractModeBasis, f::PropagationFactors, direction::Direction,
                         ψ::Tuple, stored::Tuple)
    Ψ = SVector(ψ)
    exp_dir, exp_opp = propagation_factors(f, direction)
    a_dir = exp_dir .* decompose(m, Ψ, direction)
    a_opp = exp_opp .* SVector(stored)
    Tuple(recompose(m, a_dir, direction) +
        recompose(m, a_opp, reverse(direction))), Tuple(a_dir)
end

function propagate_modes(m::AbstractModeBasis, f::PropagationFactors, direction::Direction,
                         ψ::Tuple, ::Tuple{})
    Ψ = SVector(ψ)
    exp_dir, exp_opp = propagation_factors(f, direction)
    a_dir = exp_dir .* decompose(m, Ψ, direction)
    a_opp = exp_opp .* decompose(m, Ψ, reverse(direction))
    Tuple(recompose(m, a_dir, direction) + recompose(m, a_opp, reverse(direction))), ()
end

function propagate_modes_adjoint(m::AbstractModeBasis, f::PropagationFactors,
                                 direction::Direction, ∂ψ::Tuple, ∂stored::Tuple)
    ∂Ψ = SVector(∂ψ)
    exp_dir, exp_opp = propagation_factors(f, direction)
    ∂a_dir = conj.(exp_dir) .* (recompose_adjoint(m, ∂Ψ, direction) + SVector(∂stored))
    ∂a_opp = conj.(exp_opp) .* recompose_adjoint(m, ∂Ψ, reverse(direction))
    Tuple(decompose_adjoint(m, ∂a_dir, direction)), Tuple(∂a_opp)
end

function propagate_modes_adjoint(m::AbstractModeBasis, f::PropagationFactors,
                                 direction::Direction, ∂ψ::Tuple, ::Tuple{})
    ∂Ψ = SVector(∂ψ)
    exp_dir, exp_opp = propagation_factors(f, direction)
    ∂a_dir = conj.(exp_dir) .* recompose_adjoint(m, ∂Ψ, direction)
    ∂a_opp = conj.(exp_opp) .* recompose_adjoint(m, ∂Ψ, reverse(direction))
    Tuple(decompose_adjoint(m, ∂a_dir, direction) +
        decompose_adjoint(m, ∂a_opp, reverse(direction))), ()
end

@kernel function modal_kernel!(f, fields, stored, coefs, direction)
    I = @index(Global, Cartesian)
    c = map(a -> a[I], coefs)
    for J in CartesianIndices(axes(first(fields))[3:end])
        ψ = map(a -> a[I, J], fields)
        a_st = map(a -> a[I, J], stored)
        ψ_out, a_out = f(c..., direction, ψ, a_st)
        map((a, v) -> a[I, J] = v, fields, ψ_out)
        map((a, v) -> a[I, J] = v, stored, a_out)
    end
end

function launch_modal!(f, fields::Tuple, stored::Tuple, coefs::Tuple, direction::Direction)
    modal_kernel!(get_backend(first(fields)))(
        f, fields, stored, coefs, direction; ndrange = size(first(fields))[1:2])
end

include("scalar_wave_propagator.jl")
include("vectorial_propagator.jl")
