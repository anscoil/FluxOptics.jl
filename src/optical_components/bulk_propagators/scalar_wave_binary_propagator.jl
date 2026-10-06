struct ScalarWaveBiProp{M, T, A, E, B, F, U, P} <: AbstractBinaryPropagator{M}
    trainability::Val{M}
    mask_xyz::A
    mask_eps::E
    eps_1::Complex{T}
    eps_2::Complex{T}
    dz::T
    medium_1::B
    medium_2::B
    factors_1::F
    factors_2::F
    u_tmp::U
    p_f::P
    nrm_f::T
end

function ScalarWaveBiProp(u::ScalarWaveField, thickness::Real,
                          mask_xyz::AbstractArray{<:Number, 3}, n1::Number, n2::Number;
                          mask_eps = nothing)
    T = real(eltype(u.E))
    ns = size(u)[1:2]
    n_slices = size(mask_xyz, 3)
    @assert size(mask_xyz)[1:2] == ns
    @assert n_slices >= 1
    dz = T(thickness / n_slices)
    eps_1, eps_2 = Complex{T}(n1)^2, Complex{T}(n2)^2
    mask, eps_xyz = binary_maps(u.E, mask_xyz, mask_eps, eps_1, eps_2)
    medium_1, medium_2 = ScalarMediumModes(u, n1), ScalarMediumModes(u, n2)
    factors_1 = PropagationFactors.(medium_1.modes, dz, true)
    factors_2 = PropagationFactors.(medium_2.modes, dz, true)
    p_f, _ = make_fft_plans(similar(u.E), (1, 2); normalize = false)
    ScalarWaveBiProp(Val(Static), mask, eps_xyz, eps_1, eps_2, dz, medium_1, medium_2,
                     factors_1, factors_2, similar(u), p_f, T(1 / prod(ns)))
end
