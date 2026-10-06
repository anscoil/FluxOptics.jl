struct ScalarWaveBPM{M, T, N, P, E, F} <: AbstractBidirectionalComponent{M}
    trainability::Val{M}
    n_xyz::N
    n0::Complex{T}
    n0_loc::Complex{T}
    dz::T
    n_sub::Int
    p_f::P
    medium::E
    medium_loc::E
    factors::F
    factors_loc::F
    conjugate::Bool
    nrm_f::T
end

Functors.@functor ScalarWaveBPM (n_xyz,)

function ScalarWaveBPM(u::ScalarWaveField, thickness::Real,
                       n_xyz::AbstractArray{<:Number, 3}, n0::Number;
                       n_sub::Integer = 1, n0_loc::Number = real(n0),
                       trainable::Bool = false, conjugate::Bool = false)
    T = real(eltype(u.E))
    ns = size(u)[1:2]
    n_slices = size(n_xyz, 3)
    @assert size(n_xyz)[1:2] == ns
    @assert n_slices >= 1 && n_sub >= 1
    dz = T(thickness / (n_slices * n_sub))
    N = isreal(n_xyz) ? T : Complex{T}
    n_xyz_buf = similar(u.E, N, size(n_xyz))
    copyto!(n_xyz_buf, n_xyz)
    p_f, _ = make_fft_plans(similar(u.E), (1, 2); normalize = false)
    medium = ScalarMediumModes(u, n0)
    medium_loc = ScalarMediumModes(u, n0_loc)
    factors = PropagationFactors.(medium.modes, dz, conjugate)
    factors_loc = PropagationFactors.(medium_loc.modes, dz, conjugate)
    M = trainable ? Trainable : Static
    ScalarWaveBPM(Val(M), n_xyz_buf, Complex{T}(n0), Complex{T}(n0_loc), dz, n_sub, p_f,
                  medium, medium_loc, factors, factors_loc, conjugate, T(1 / prod(ns)))
end

trainable(p::ScalarWaveBPM{Trainable}) = (; n_xyz = p.n_xyz)

reference_medium(p::ScalarWaveBPM) = p.medium

function alloc_fp_state(u::ScalarWaveField, p::ScalarWaveBPM)
    p.conjugate ? nothing : (; amp = similar(u.E, (size(u.E)..., size(p.n_xyz, 3))))
end

function alloc_activations(u::ScalarWaveField, p::ScalarWaveBPM, ::Direction)
    (; u = similar(u.E, (size(u.E)..., size(p.n_xyz, 3))))
end

slice_at(a::AbstractArray, k::Integer) = selectdim(a, ndims(a), k)

function kick_coefficient(u::ScalarWaveField, p::ScalarWaveBPM, direction::Direction)
    sign(direction) * real(eltype(u.E))((2π / u.lambda)^2) * p.dz
end

function kick(E, dzE, n_xy, n0, c, nrm)
    E_n = nrm * E
    dzE_n = nrm * dzE + c * (n0^2 - n_xy^2) * E_n
    E_n, dzE_n
end

function kick_and_store(E, dzE, n_xy, n0, c, nrm)
    E_n, dzE_n = kick(E, dzE, n_xy, n0, c, nrm)
    E_n, dzE_n, E_n
end

function kick_adjoint(∂E, ∂dzE, n_xy, n0, c, nrm)
    nrm * (∂E + c * conj(n0^2 - n_xy^2) * ∂dzE), nrm * ∂dzE
end

function compute_gradient!(∂n_xy, n_xy, u_act, ∂u::ScalarWaveField, c)
    @. ∂n_xy = -2c * real(conj(n_xy * u_act) * ∂u.dzE)
end

function propagate_slice_fourier!(u::ScalarWaveField, state, p::ScalarWaveBPM,
                                  k::Integer, direction::Direction; loc::Bool = false)
    medium, factors = loc ? (p.medium_loc, p.factors_loc) : (p.medium, p.factors)
    stored = map(a -> slice_at(a, k), state_arrays(state))
    launch_modal!(propagate_modes, (u.E, u.dzE), stored, (medium.modes, factors), direction)
end

function propagate_slice_direct!(u::ScalarWaveField, activations, p::ScalarWaveBPM,
                                 k::Integer, direction::Direction; loc::Bool = false)
    n_xy = view(p.n_xyz, :, :, k)
    n0 = loc ? p.n0_loc : p.n0
    c = kick_coefficient(u, p, direction)
    compute_ift!(p.p_f, u)
    if isnothing(activations)
        StructArray((u.E, u.dzE)) .= kick.(u.E, u.dzE, n_xy, n0, c, p.nrm_f)
    else
        StructArray((u.E, u.dzE, slice_at(activations.u, k))) .=
            kick_and_store.(u.E, u.dzE, n_xy, n0, c, p.nrm_f)
    end
    compute_ft!(p.p_f, u)
end

function propagate_slice!(u::ScalarWaveField, state, activations,
                          p::ScalarWaveBPM, k::Integer, ::Forward; loc::Bool = false)
    propagate_slice_direct!(u, activations, p, k, Forward(); loc)
    propagate_slice_fourier!(u, state, p, k, Forward(); loc)
    u
end

function propagate_slice!(u::ScalarWaveField, state, activations,
                          p::ScalarWaveBPM, k::Integer, ::Backward; loc::Bool = false)
    propagate_slice_fourier!(u, state, p, k, Backward(); loc)
    propagate_slice_direct!(u, activations, p, k, Backward(); loc)
    u
end

function propagate_slice_adjoint_fourier!(∂u::ScalarWaveField, state, p::ScalarWaveBPM,
                                          k::Integer, direction::Direction)
    stored = map(a -> slice_at(a, k), state_arrays(state))
    launch_modal!(propagate_modes_adjoint, (∂u.E, ∂u.dzE), stored,
                  (p.medium.modes, p.factors), direction)
end

function propagate_slice_adjoint_direct!(∂u::ScalarWaveField, ∂p, activations,
                                         p::ScalarWaveBPM, k::Integer, direction::Direction)
    n_xy = view(p.n_xyz, :, :, k)
    c = kick_coefficient(∂u, p, direction)
    compute_ift!(p.p_f, ∂u)
    if !isnothing(activations)
        compute_gradient!(slice_at(∂p.n_xyz, k), n_xy, slice_at(activations.u, k), ∂u, c)
    end
    StructArray((∂u.E, ∂u.dzE)) .= kick_adjoint.(∂u.E, ∂u.dzE, n_xy, p.n0, c, p.nrm_f)
    compute_ft!(p.p_f, ∂u)
end

function propagate_slice_adjoint!(∂u::ScalarWaveField, ∂p, state, activations,
                                  p::ScalarWaveBPM, k::Integer, ::Forward)
    propagate_slice_adjoint_fourier!(∂u, state, p, k, Forward())
    propagate_slice_adjoint_direct!(∂u, ∂p, activations, p, k, Forward())
    ∂u
end

function propagate_slice_adjoint!(∂u::ScalarWaveField, ∂p, state, activations,
                                  p::ScalarWaveBPM, k::Integer, ::Backward)
    propagate_slice_adjoint_direct!(∂u, ∂p, activations, p, k, Backward())
    propagate_slice_adjoint_fourier!(∂u, state, p, k, Backward())
    ∂u
end

function propagate!(u::ScalarWaveField, state, activations, p::ScalarWaveBPM,
                    direction::Direction)
    for k in reverse(1:size(p.n_xyz, 3), direction)
        for j in reverse(1:p.n_sub, direction)
            if j == 1
                propagate_slice!(u, state, activations, p, k, direction)
            else
                propagate_slice!(u, nothing, nothing, p, k, direction; loc = true)
            end
        end
    end
    u
end

function propagate_adjoint!(u::ScalarWaveField, ∂p, state, activations, p::ScalarWaveBPM,
                            direction::Direction)
    p.n_sub == 1 || throw(ArgumentError(
        "ScalarWaveBPM: gradients require n_sub = 1 \
        (n_sub > 1 is meant for forward convergence checks)"))
    for k in reverse(1:size(p.n_xyz, 3), reverse(direction))
        propagate_slice_adjoint!(u, ∂p, state, activations, p, k, direction)
    end
    u
end
