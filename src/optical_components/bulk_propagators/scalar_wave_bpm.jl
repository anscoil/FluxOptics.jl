struct ScalarWaveBPM{M, T, N, P, E, F, R, Q} <: AbstractBidirectionalComponent{M}
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
    transports::R        # nothing: the opposite modes are kept in the state
    transports_loc::Q    # stateless sub-steps (n_sub > 1); nothing for n_sub = 1
    nrm_f::T
end

Functors.@functor ScalarWaveBPM (n_xyz,)

"""
    ScalarWaveBPM(u, thickness, n_xyz, n0; n_sub = 1, n0_loc = real(n0),
                  trainable = false, stateful = true, keep_evanescent = false)

Bidirectional scalar split-step propagation through a layer of given thickness. A
stateful BPM keeps the modes travelling against each pass in the fixed-point state; a
stateless one transports them across each step, cutting the evanescent ones unless
keep_evanescent. The n_sub - 1 sub-steps of each slice are always stateless.
"""
function ScalarWaveBPM(u::ScalarWaveField, thickness::Real,
                       n_xyz::AbstractArray{<:Number, 3}, n0::Number;
                       n_sub::Integer = 1, n0_loc::Number = real(n0),
                       trainable::Bool = false, stateful::Bool = true,
                       keep_evanescent::Bool = false)
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
    medium_loc = ScalarMediumModes(u, Complex{T}(n0_loc))
    factors = propagation_factors.(medium.modes, dz)
    factors_loc = propagation_factors.(medium_loc.modes, dz)
    transports = stateful ? nothing : transport_factors.(medium.modes, dz; keep_evanescent)
    # Sub-steps are short: with a state, their opposite modes are transported exactly
    keep_loc = stateful || keep_evanescent
    transports_loc = n_sub == 1 ? nothing :
        transport_factors.(medium_loc.modes, dz; keep_evanescent = keep_loc)
    M = trainable ? Trainable : Static
    ScalarWaveBPM(Val(M), n_xyz_buf, Complex{T}(n0), Complex{T}(n0_loc), dz, n_sub, p_f,
                  medium, medium_loc, factors, factors_loc, transports, transports_loc,
                  T(1 / prod(ns)))
end

trainable(p::ScalarWaveBPM{Trainable}) = (; n_xyz = p.n_xyz)

reference_medium(p::ScalarWaveBPM) = p.medium

function alloc_fp_state(u::ScalarWaveField, p::ScalarWaveBPM)
    isnothing(p.transports) ? (; amp = similar(u.E, (size(u.E)..., size(p.n_xyz, 3)))) :
        nothing
end

function alloc_activations(u::ScalarWaveField, p::ScalarWaveBPM, ::Direction)
    (; u = similar(u.E, (size(u.E)..., size(p.n_xyz, 3))))
end

function kick_coefficient(u::ScalarWaveField, dz, direction::Direction)
    sign(direction) * real(eltype(u.E))((2π / u.lambda)^2) * dz
end

function kick(ψ::ScalarWaveState, w, ε, ε_ref, c)
    E = w * ψ.E
    ScalarWaveState(E, w * ψ.dzE + c * (ε_ref - ε) * E)
end

function kick_adjoint(∂ψ::ScalarWaveState, w, ε, ε_ref, c)
    ∂E = conj(w) * (∂ψ.E + conj(c * (ε_ref - ε)) * ∂ψ.dzE)
    ScalarWaveState(∂E, conj(w) * ∂ψ.dzE)
end

function kick_and_store(ψ::ScalarWaveState, w, ε, ε_ref, c)
    ψ_out = kick(ψ, w, ε, ε_ref, c)
    ψ_out, ψ_out.E
end

function compute_gradient!(∂n_xy, n_xy, u_act, ∂u::ScalarWaveField, c)
    @. ∂n_xy = -2c * real(conj(n_xy * u_act) * ∂u.dzE)
end

# The sub-steps (loc) are stateless: state is nothing and their transports are set
function propagate_slice_fourier!(u::ScalarWaveField, state, p::ScalarWaveBPM,
                                  k::Integer, direction::Direction; loc::Bool = false)
    stored = map(a -> slice_at(a, k), state_arrays(state))
    if loc
        coefs = modal_coefs(p.medium_loc, p.factors_loc, p.transports_loc)
        launch_modal!(propagate_modes, (u.E, u.dzE), stored, coefs, direction)
    else
        coefs = modal_coefs(p.medium, p.factors, p.transports)
        launch_modal!(propagate_modes, (u.E, u.dzE), stored, coefs, direction)
    end
end

function propagate_slice_direct!(u::ScalarWaveField, activations, p::ScalarWaveBPM,
                                 k::Integer, direction::Direction; loc::Bool = false)
    n_xy = slice_at(p.n_xyz, k)
    n0 = loc ? p.n0_loc : p.n0
    c = kick_coefficient(u, p.dz, direction)
    ψ = StructArray(u)
    compute_ift!(p.p_f, u)
    if isnothing(activations)
        ψ .= kick.(ψ, p.nrm_f, n_xy .^ 2, n0^2, c)
    else
        StructArray((ψ, slice_at(activations.u, k))) .=
            kick_and_store.(ψ, p.nrm_f, n_xy .^ 2, n0^2, c)
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
                  modal_coefs(p.medium, p.factors, p.transports), direction)
end

function propagate_slice_adjoint_direct!(∂u::ScalarWaveField, ∂p, activations,
                                         p::ScalarWaveBPM, k::Integer, direction::Direction)
    n_xy = slice_at(p.n_xyz, k)
    c = kick_coefficient(∂u, p.dz, direction)
    ∂ψ = StructArray(∂u)
    compute_ift!(p.p_f, ∂u)
    if !isnothing(activations)
        compute_gradient!(slice_at(∂p.n_xyz, k), n_xy, slice_at(activations.u, k), ∂u, c)
    end
    ∂ψ .= kick_adjoint.(∂ψ, p.nrm_f, n_xy .^ 2, p.n0^2, c)
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
