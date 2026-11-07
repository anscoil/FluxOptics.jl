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
    medium_loc = ScalarMediumModes(u, n0_loc)
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

function alloc_fp_state(u::ScalarWaveField, p::ScalarWaveBPM)
    isnothing(p.transports) ? (; amp = similar(u.E, (size(u.E)..., size(p.n_xyz, 3)))) :
        nothing
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

function propagate_slice_adjoint_fourier!(∂u::ScalarWaveField, state, p::ScalarWaveBPM,
                                          k::Integer, direction::Direction)
    stored = map(a -> slice_at(a, k), state_arrays(state))
    launch_modal!(propagate_modes_adjoint, (∂u.E, ∂u.dzE), stored,
                  modal_coefs(p.medium, p.factors, p.transports), direction)
end
