struct VectorialBPM{M, T, N, R, P, E, F, K, B, S, A, Q} <:
       AbstractBidirectionalComponent{M}
    trainability::Val{M}
    eps_xyz::N
    eps_ref::R
    dz::T
    p_f::P             # FFT plans: of one field (field) and of the pair buffer (pair)
    medium::E
    factors::F         # modal step over dz
    half_factors::F    # modal step over dz/2: first and last steps with strang
    kx::K
    ky::K
    buffers::B         # pair buffer of the kick and its components, δ of L
    storage::S         # stored planes and interpolation weights, per pass (fwd, bwd)
    forward_amps::A    # forward amplitudes of the stored slices, for the backward pass
    phasors::Q         # interaction picture: phase factors per pixel, or nothing
    symmetric::Bool    # Strang splitting of the sheets: L/2, K, L/2
    strang::Bool       # half modal steps at both ends of the layer (see Stages)
    nrm_f::T
end

Functors.@functor VectorialBPM (eps_xyz,)

# ---------------------------------------------------------------------------------------
# Permittivity maps
#
# eps_xyz holds one permittivity per voxel: a Number (isotropic medium) or a ZDecEpsilon
# (z-decoupled tensor: in-plane 2×2 block and εzz, no xz, yz, zx, zy terms), stored on
# the device of the field and in its precision. The kernels dispatch on the type of the
# voxel.
#
# General tensors (Epsilon) are not handled: their xz, yz, zx, zy terms couple E⊥ to
# itself and H⊥ to itself through Ez, which are not shears, so the exact inverse of
# the backward pass no longer holds; they need the scheme with auxiliary δ variables.
# ---------------------------------------------------------------------------------------

# Element type of a permittivity map in the precision T of the field
map_eltype(::Type{<:Real}, ::Type{T}) where {T} = T
map_eltype(::Type{<:Complex}, ::Type{T}) where {T} = Complex{T}
map_eltype(::Type{ZDecEpsilon{S}}, ::Type{T}) where {S, T} = ZDecEpsilon{map_eltype(S, T)}

# Tensor maps are stored one array per component (StructArray): the kernels then read
# contiguous arrays of scalars
struct_of_arrays(eps::AbstractArray{<:Number}) = eps
struct_of_arrays(eps::AbstractArray{<:ZDecEpsilon}) = StructArray(eps)

# Kick coefficients: the permittivity slice (isotropic) or its four in-plane component
# arrays (ZDecEpsilon), so that the loop only indexes plain arrays
kick_coefs(ε::AbstractArray{<:Number}) = (ε,)
kick_coefs(ε::StructArray{<:ZDecEpsilon}) = (ε.xx, ε.xy, ε.yx, ε.yy)

# The longitudinal sheet only reads εzz: pass it the zz array alone
zz_component(ε::AbstractArray{<:Number}) = ε
zz_component(ε::StructArray{<:ZDecEpsilon}) = ε.zz

# ---------------------------------------------------------------------------------------
# Stored slices
#
# The state holds the opposite amplitudes on a subset of the slices only, spread evenly
# from the first slice to the last one, at most `step` slices apart. At the other slices,
# the opposite amplitudes are rebuilt by cubic Lagrange interpolation in z from the four
# nearest stored slices. Even spacing keeps the weights small: a short last interval
# would raise their sum of moduli up to about 9. The weights are fixed, so the round
# trip stays linear. step = 1 stores every slice.
#
# Interaction picture (interaction = true). In a homogeneous medium the amplitudes
# carry the phase of free propagation, exp(±i·kz·z) for each mode, and interpolating
# them as they are interpolates this carrier, with an error of about 0.023·θ₀⁴ for a
# phase θ₀ between two stored slices. The state then holds the envelopes
# b = a·exp(-i·σ·k·θ) instead, θ = Re(kz)·dz per pixel, k the slice and σ = ±1 the
# direction of the pass that stores them; the opposite pass reads a = b·exp(-i·σ'·k·θ)
# with its own σ' = -σ. The envelopes are constant in a homogeneous medium, so that
# their interpolation is exact there, and only vary through the kicks. Only the phase
# is removed, never the decay exp(-Im(kz)·z): the factors have unit modulus, so the
# change of variables is unitary (same Krylov iterations) and cannot amplify the decay
# of absorbed or evanescent modes, whose interpolation is unchanged. The factor
# exp(-i·σ·k·θ) is the same for storing and reading during a pass; it is updated slice
# after slice by one product with exp(-i·θ), whatever the direction.
#
# The interpolation and the storage are part of the modal step kernel (see Stages).
# ---------------------------------------------------------------------------------------

struct SliceStorage{T}
    slots::Vector{Int}                # slot of each slice in the state, 0 if not stored
    nodes::Vector{NTuple{4, Int}}     # slots of the interpolation nodes of each slice
    weights::Vector{NTuple{4, T}}     # their cubic Lagrange weights
end

# positions: z of the planes, in any unit (uniform by default). With strang, the planes
# of the two passes are not evenly spaced at the ends, and each pass has its own weights.
function SliceStorage(::Type{T}, nz::Integer, step::Integer;
                      positions::AbstractVector{<:Real} = 1:nz) where {T}
    length(positions) == nz || throw(ArgumentError("one position per plane expected"))
    step >= 1 || throw(ArgumentError("step must be at least 1"))
    n = cld(nz - 1, step) + 1
    zs = unique!(round.(Int, range(1, nz; length = n)))
    length(zs) == nz || length(zs) >= 4 ||
        throw(ArgumentError("interpolation needs at least 4 stored slices"))
    slots = zeros(Int, nz)
    slots[zs] .= eachindex(zs)
    nodes = fill(ntuple(_ -> 0, 4), nz)
    weights = fill(ntuple(_ -> zero(T), 4), nz)
    for k in 1:nz
        slots[k] > 0 && continue    # stored slices are copied, never interpolated
        # Four nearest stored slices, two on each side when possible
        j = clamp(searchsortedlast(zs, k) - 1, 1, length(zs) - 3)
        nodes[k] = ntuple(l -> j + l - 1, 4)
        z = ntuple(l -> positions[zs[j + l - 1]], 4)
        x = positions[k]
        weights[k] = ntuple(4) do l
            T(prod((x - z[m]) / (z[l] - z[m]) for m in 1:4 if m != l))
        end
    end
    SliceStorage(slots, nodes, weights)
end

n_stored(s::SliceStorage) = maximum(s.slots)

# Cubic interpolation of both opposite amplitudes: src holds the four node values of
# amp1, then those of amp2
@inline function interpolate_pair(w, src)
    w1, w2, w3, w4 = w
    a1, b1, c1, d1, a2, b2, c2, d2 = src
    (w1 * a1 + w2 * b1 + w3 * c1 + w4 * d1, w1 * a2 + w2 * b2 + w3 * c2 + w4 * d2)
end

# Phase factor outside the interaction picture: one for every pixel, which the products
# P * a fold away
struct Unity end
@inline Base.getindex(::Unity, ::Vararg{Any}) = true

# Calls f with the description of slice k for its modal step:
# - sources: where its opposite amplitudes come from, the slice itself if it is stored
#   (weights = nothing, a copy) or the four nodes of each amplitude (interpolation);
# - targets: its slot in the departing arrays if it is stored, () otherwise;
# - phasor: the phase factor of the pass (Unity outside the interaction picture).
# Each branch has its own concrete types, so the slice loop dispatches statically.
function with_slice(f, s::SliceStorage, k::Integer, opposite::Tuple, departing::Tuple,
                    P)
    j = s.slots[k]
    if j > 0
        f((; weights = nothing, sources = map(b -> slice_at(b, j), opposite),
           targets = map(b -> slice_at(b, j), departing), phasor = P))
    else
        n = s.nodes[k]
        a, b = opposite
        sources = (map(i -> slice_at(a, i), n)..., map(i -> slice_at(b, i), n)...)
        f((; weights = s.weights[k], sources, targets = (), phasor = P))
    end
end

"""
    VectorialBPM(u, thickness, eps_xyz, eps_ref; step = 1, interaction = false,
                 symmetric = false, strang = false, trainable = false)

Bidirectional vectorial split-step propagation through a layer of given thickness.
`eps_xyz` has shape (Nx, Ny, nz), one slice per z step, with Number (isotropic) or
ZDecEpsilon (z-decoupled tensor) entries. `eps_ref` is the scalar permittivity of the
reference medium in which the modal propagation is exact. The state stores slices spread
evenly from the first to the last, at most `step` slices apart, and interpolates the
others (step = 1 stores every slice); with `interaction = true`, it stores envelopes,
free propagation phase removed, so that the interpolation is exact in homogeneous media
(see Stored slices). With `symmetric = true`, each slice applies L/2, K, L/2 instead of
K, L (see Stages). With `strang = true`, the layer also starts and ends with a half modal
step, which makes the scheme of second order in dz (Strang splitting of the whole slice
sequence, sheets symmetric); it needs one more stored plane and no interaction picture.
"""
function VectorialBPM(u::VectorialField, thickness::Real,
                      eps_xyz::AbstractArray{<:Permittivity, 3}, eps_ref::Number;
                      step::Integer = 1, interaction::Bool = false,
                      symmetric::Bool = false, strang::Bool = false,
                      trainable::Bool = false)
    eltype(eps_xyz) <: Epsilon && throw(ArgumentError(
        "VectorialBPM handles isotropic and z-decoupled permittivities (ZDecEpsilon); " *
            "general tensors couple E⊥ and Ez and need the δ-variable scheme"))
    T = real(eltype(u.Ex))
    ns = size(u.Ex)[1:2]
    @assert size(eps_xyz)[1:2] == ns
    nz = size(eps_xyz, 3)
    dz = T(thickness / nz)
    strang && interaction && throw(ArgumentError(
        "strang = true does not support the interaction picture (interaction = true)"))
    symmetric = symmetric || strang
    # Conversion on the host first, so that the copy to the device never mixes types
    E = map_eltype(eltype(eps_xyz), T)
    eps_buf = adapt(get_backend(u.Ex), struct_of_arrays(collect(E, eps_xyz)))
    ε_r = isreal(eps_ref) ? T(real(eps_ref)) : Complex{T}(eps_ref)
    # Kick buffer: E⊥ components stacked along a last dimension, so that both are
    # transformed by one batched FFT, the plan seeing it as (Nx, Ny, batch·2)
    pair = similar(u.Ex, (size(u.Ex)..., 2))
    buffers = (; pair = reshape(pair, ns..., :), x = selectdim(pair, ndims(pair), 1),
               y = selectdim(pair, ndims(pair), 2), delta = similar(u.Ex))
    p_f = (; field = first(make_fft_plans(similar(u.Ex), (1, 2); normalize = false)),
           pair = first(make_fft_plans(buffers.pair, (1, 2); normalize = false)))
    medium = VectorialMediumModes(u, ε_r)
    factors = propagation_factors.(medium.modes, dz)
    half_factors = propagation_factors.(medium.modes, dz / 2)
    fx = collect(T, T(2π) .* fftfreq(ns[1], 1 / u.ds[1]))
    fy = collect(T, T(2π) .* fftfreq(ns[2], 1 / u.ds[2]))
    kx, ky = similar(u.Ex, T, ns), similar(u.Ex, T, ns)
    copyto!(kx, repeat(fx, 1, ns[2]))
    copyto!(ky, repeat(permutedims(fy), ns[1], 1))
    storage = plane_storage(T, nz, step, strang)
    dims = (size(u.Ex)..., n_stored(storage.fwd))
    forward_amps = (similar(u.Ex, dims), similar(u.Ex, dims))
    # Interaction picture: exp(-i·θ) per pixel, θ = Re(kz)·dz, and the phase factor
    # exp(-i·σ·k·θ) at the first slice of each pass: k = 1 (σ = 1), k = nz (σ = -1)
    phasors = if interaction
        θ = map(m -> T(real(m.kz)) * dz, medium.modes)
        (; step = cis.(-θ), first_fwd = cis.(-θ), first_bwd = cis.(nz .* θ),
         current = similar(u.Ex, Complex{T}, ns))
    else
        nothing
    end
    M = trainable ? Trainable : Static
    VectorialBPM(Val(M), eps_buf, ε_r, dz, p_f, medium, factors, half_factors, kx, ky,
                 buffers, storage, forward_amps, phasors, symmetric, strang,
                 T(1 / prod(ns)))
end

# Planes of the modal steps, in units of dz. Without strang: the step of slice k goes from
# k - 1 to k. With strang: step 0 goes from 0 to 1/2, step k from k - 1/2 to k + 1/2 for
# 1 ≤ k < nz, and step nz from nz - 1/2 to nz; step m uses plane m + 1 of the storage.
# The forward pass reads the backward amplitudes at the end of each step, the backward
# pass the forward amplitudes at its start: the interpolation weights of each pass use
# the positions of the planes it reads.
function plane_storage(::Type{T}, nz, step, strang::Bool) where {T}
    if strang
        ends = vcat((0:(nz - 1)) .+ 0.5, nz)
        starts = vcat(0, (1:nz) .- 0.5)
        (; fwd = SliceStorage(T, nz + 1, step; positions = ends),
         bwd = SliceStorage(T, nz + 1, step; positions = starts))
    else
        s = SliceStorage(T, nz, step)
        (; fwd = s, bwd = s)
    end
end

reference_medium(p::VectorialBPM) = p.medium

function alloc_fp_state(u::VectorialField, p::VectorialBPM)
    dims = (size(u.Ex)..., n_stored(p.storage.fwd))
    (; amp1 = similar(u.Ex, dims), amp2 = similar(u.Ex, dims))
end

# ---------------------------------------------------------------------------------------
# One slice applies three operators, each over the full step dz (Lie splitting):
#   K (kick):         H⊥ += cK·J·[(ε⊥ - ε_r)·E⊥],  J(Dx, Dy) = (-Dy, Dx)
#   L (longitudinal): E⊥ += k⊥·[cL·(Δεzz/εzz)·D̂z],  D̂z = -(kx·Hy - ky·Hx)/k0
#   P (propagation):  modal propagation in the reference medium ε_r
# with ε⊥ the in-plane 2×2 block of the permittivity (ε·I for isotropic media) and
# Δεzz = εzz - ε_r. The products are pointwise on the grid (collocation), with all the
# frequencies of the grid.
# K and L are shears (one pair of components updated from the other), so their inverse
# is the same operator with the opposite coefficient: sign(direction) in cK and cL.
# Forward: K, L, P. Backward: P, L, K, the exact inverse sequence.
#
# symmetric = true: L/2, K, L/2 (Strang splitting of the sheets), which removes the
# first order error of the commutator [K, L]. The sequence is a palindrome, the same in
# both passes. L/2 is still a shear, and L/2∘L/2 = L. Alone, it keeps the scheme of
# first order: the sheets stay at the start of each slice, which moves a layer dz/2 of
# reference medium from one end of the layer to the other.
#
# strang = true (implies symmetric): the layer starts with a modal step over dz/2 and
# its last modal step is over dz/2, so that the sheets sit in the middle of their slices:
#   P/2, [L/2, K, L/2, P] × (nz - 1), L/2, K, L/2, P/2
# a Strang splitting of the whole sequence, of second order in dz. The first half step
# is one more modal step, with its own stored plane (plane 1; slice k uses plane k + 1).
#
# Stages. K and L are sheets with the same pattern, input and output being pointwise in
# Fourier space:
#   input → IFFT → product (real space) → FFT → output
#   K: kick_input (pair = E⊥), kick_sheet, kick_output (H⊥ += ΔH⊥)
#   L: longitudinal_input (δ = D̂z), longitudinal_sheet, longitudinal_output (E⊥ += k⊥·δ)
# P is a single pointwise modal_step, which also reads the opposite amplitudes (copied or
# interpolated from the state) and writes the departing ones if the slice is stored.
# Wherever the output of an operator meets the input of the next one, both run in one
# kernel, one pass over the grid instead of two:
#   kick_output + longitudinal_input        longitudinal_output + kick_input
#   longitudinal_output + modal_step        modal_step + longitudinal_input
# The kick works in the pair buffer (E⊥ stacked along a last dimension, one batched FFT
# for both components), the longitudinal sheet in δ.
#
# Every pixel function is inlined into the loops of launch_combine!: a call left out of
# line breaks their @simd vectorization on CPU (about ten times slower).
# ---------------------------------------------------------------------------------------

# Pointwise pieces, Fourier space
@inline added_kick(H, ΔH) = (H[1] + ΔH[1], H[2] + ΔH[2])
@inline normalized_dz(k0, kx, ky, H) = -(kx * H[2] - ky * H[1]) / k0      # D̂z
@inline added_longitudinal(kx, ky, E, δ) = (E[1] + kx * δ, E[2] + ky * δ)

# Kick product, real space: the pair buffer holds E⊥ on input and ΔH⊥ on output,
# ΔHx = -c·ΔDy and ΔHy = c·ΔDx with ΔD = (ε⊥ - ε_r)·E⊥
struct KickProduct{C, R}
    c::C
    r::R
end

@inline function (op::KickProduct)(ε, ::Direction, ψ, ψ_in)
    Ex, Ey = ψ
    Δ = ε - op.r
    (-op.c * Δ * Ey, op.c * Δ * Ex)
end

@inline function (op::KickProduct)(εxx, εxy, εyx, εyy, ::Direction, ψ, ψ_in)
    Ex, Ey = ψ
    Dx, Dy = εxx * Ex + εxy * Ey, εyx * Ex + εyy * Ey
    (-op.c * (Dy - op.r * Ey), op.c * (Dx - op.r * Ex))
end

# Longitudinal product, real space: δ = c·(Δεzz/εzz)·D̂z, i.e. c·Δεzz·Ez
struct LongitudinalProduct{C, R}
    c::C
    r::R
end

@inline function (op::LongitudinalProduct)(εzz, ::Direction, ψ, ψ_in)
    δ, = ψ
    (op.c * δ * (εzz - op.r) / εzz,)
end

# Kick input: pair = E⊥ (fields: pair, inputs: E⊥)
@inline copy_inputs(::Direction, ψ, ψ_in) = ψ_in

# Kick output: H⊥ += ΔH⊥ (fields: H⊥, inputs: pair)
@inline kick_output(::Direction, ψ, ψ_in) = added_kick(ψ, ψ_in)

# Longitudinal input: δ = D̂z (fields: δ, inputs: H⊥)
struct LongitudinalInput{T}
    k0::T
end

@inline function (op::LongitudinalInput)(kx, ky, ::Direction, ψ, ψ_in)
    (normalized_dz(op.k0, kx, ky, ψ_in),)
end

# Longitudinal output: E⊥ += k⊥·δ (fields: E⊥, inputs: δ)
@inline function longitudinal_output(kx, ky, ::Direction, ψ, ψ_in)
    added_longitudinal(kx, ky, ψ, ψ_in[1])
end

# Kick output, then longitudinal input (fields: H⊥, δ; inputs: pair)
struct KickOutputLongitudinalInput{T}
    k0::T
end

@inline function (op::KickOutputLongitudinalInput)(kx, ky, ::Direction, ψ, ψ_in)
    H = added_kick(ψ, ψ_in)
    (H..., normalized_dz(op.k0, kx, ky, H))
end

# Longitudinal output, then kick input (fields: E⊥, pair; inputs: δ)
@inline function longitudinal_output_kick_input(kx, ky, ::Direction, ψ, ψ_in)
    E = added_longitudinal(kx, ky, ψ, ψ_in[1])
    (E..., E...)
end

# Modal step of a pixel: the opposite amplitudes are rebuilt from their sources, a copy
# (weights = nothing) or the interpolation of four nodes, the departing ones returned
# for storage if the slice is stored. Both are envelopes times the phase factor P.
struct ModalStep{W, S}
    weights::W
    store::Val{S}
end

stores(::Tuple{}) = Val(false)
stores(::Tuple) = Val(true)
ModalStep(slice::NamedTuple) = ModalStep(slice.weights, stores(slice.targets))

@inline opposite_pair(::Nothing, P, src) = (P * src[1], P * src[2])
@inline function opposite_pair(w::NTuple{4}, P, src)
    a, b = interpolate_pair(w, src)
    (P * a, P * b)
end

@inline departing_pair(::Val{true}, P, a) = (P * a[1], P * a[2])
@inline departing_pair(::Val{false}, P, a) = ()

# Ψ = (E⊥, H⊥) → (new Ψ, departing amplitudes to store)
@inline function (op::ModalStep)(m, f, P, direction::Direction, Ψ, src)
    o = opposite_pair(op.weights, P, src)
    Ψ_out, a = @inline propagate_modes(m, f, direction, Ψ, o)
    (Ψ_out, departing_pair(op.store, P, a))
end

# Longitudinal output, then modal step (fields: E⊥, H⊥, targets; inputs: δ, sources)
struct LongitudinalOutputModalStep{S}
    step::S
end

@inline function (op::LongitudinalOutputModalStep)(kx, ky, m, f, P, direction::Direction,
                                                   ψ, ψ_in)
    Ex, Ey, Hx, Hy = ψ
    E = added_longitudinal(kx, ky, (Ex, Ey), first(ψ_in))
    Ψ, a = op.step(m, f, P, direction, (E..., Hx, Hy), Base.tail(ψ_in))
    (Ψ..., a...)
end

# Modal step, then longitudinal input (fields: E⊥, H⊥, δ, targets; inputs: sources)
struct ModalStepLongitudinalInput{S, T}
    step::S
    k0::T
end

@inline function (op::ModalStepLongitudinalInput)(kx, ky, m, f, P, direction::Direction,
                                                  ψ, ψ_in)
    Ex, Ey, Hx, Hy = ψ
    Ψ, a = op.step(m, f, P, direction, (Ex, Ey, Hx, Hy), ψ_in)
    (Ψ..., normalized_dz(op.k0, kx, ky, (Ψ[3], Ψ[4])), a...)
end

# Modal step alone, for the half steps at the ends of the layer (fields: E⊥, H⊥, targets;
# inputs: sources)
struct ModalStepOnly{S}
    step::S
end

@inline function (op::ModalStepOnly)(kx, ky, m, f, P, direction::Direction, ψ, ψ_in)
    Ex, Ey, Hx, Hy = ψ
    Ψ, a = op.step(m, f, P, direction, (Ex, Ey, Hx, Hy), ψ_in)
    (Ψ..., a...)
end

# ---------------------------------------------------------------------------------------
# Stages: one launch each, plus two FFTs for each sheet
# ---------------------------------------------------------------------------------------

k_grid(p::VectorialBPM) = (p.kx, p.ky)
modal_coefs(p::VectorialBPM, slice) =
    (p.kx, p.ky, p.medium.modes, slice.factors, slice.phasor)

function kick_input!(u::VectorialField, p::VectorialBPM, direction::Direction)
    (; x, y) = p.buffers
    launch_combine!(copy_inputs, (x, y), (u.Ex, u.Ey), (), direction)
end

function kick_sheet!(p::VectorialBPM, ε, cK, direction::Direction)
    (; pair, x, y) = p.buffers
    p.p_f.pair.ift * pair
    launch_combine!(KickProduct(cK, p.eps_ref), (x, y), (), kick_coefs(ε), direction)
    p.p_f.pair.ft * pair
end

function kick_output!(u::VectorialField, p::VectorialBPM, direction::Direction)
    (; x, y) = p.buffers
    launch_combine!(kick_output, (u.Hx, u.Hy), (x, y), (), direction)
end

function longitudinal_input!(u::VectorialField, p::VectorialBPM, k0, direction::Direction)
    launch_combine!(LongitudinalInput(k0), (p.buffers.delta,), (u.Hx, u.Hy), k_grid(p),
                    direction)
end

function longitudinal_sheet!(p::VectorialBPM, ε, cL, direction::Direction)
    δ = p.buffers.delta
    p.p_f.field.ift * δ
    launch_combine!(LongitudinalProduct(cL, p.eps_ref), (δ,), (), (zz_component(ε),),
                    direction)
    p.p_f.field.ft * δ
end

function longitudinal_output!(u::VectorialField, p::VectorialBPM, direction::Direction)
    launch_combine!(longitudinal_output, (u.Ex, u.Ey), (p.buffers.delta,), k_grid(p),
                    direction)
end

function kick_output_longitudinal_input!(u::VectorialField, p::VectorialBPM, k0,
                                         direction::Direction)
    (; x, y, delta) = p.buffers
    launch_combine!(KickOutputLongitudinalInput(k0), (u.Hx, u.Hy, delta), (x, y),
                    k_grid(p), direction)
end

function longitudinal_output_kick_input!(u::VectorialField, p::VectorialBPM,
                                         direction::Direction)
    (; x, y, delta) = p.buffers
    launch_combine!(longitudinal_output_kick_input, (u.Ex, u.Ey, x, y), (delta,),
                    k_grid(p), direction)
end

function longitudinal_output_modal_step!(u::VectorialField, p::VectorialBPM, slice,
                                         direction::Direction)
    op = LongitudinalOutputModalStep(ModalStep(slice))
    launch_combine!(op, (u.Ex, u.Ey, u.Hx, u.Hy, slice.targets...),
                    (p.buffers.delta, slice.sources...), modal_coefs(p, slice), direction)
end

function modal_step_longitudinal_input!(u::VectorialField, p::VectorialBPM, slice, k0,
                                        direction::Direction)
    op = ModalStepLongitudinalInput(ModalStep(slice), k0)
    launch_combine!(op, (u.Ex, u.Ey, u.Hx, u.Hy, p.buffers.delta, slice.targets...),
                    slice.sources, modal_coefs(p, slice), direction)
end

function modal_step!(u::VectorialField, p::VectorialBPM, slice, direction::Direction)
    op = ModalStepOnly(ModalStep(slice))
    launch_combine!(op, (u.Ex, u.Ey, u.Hx, u.Hy, slice.targets...), slice.sources,
                    modal_coefs(p, slice), direction)
end

# ---------------------------------------------------------------------------------------
# Slice and slice loop
# ---------------------------------------------------------------------------------------

function propagate_slice!(u::VectorialField, p::VectorialBPM, ε, slice, cK, cL, k0,
                          direction::Direction)
    if p.symmetric
        # L/2, K, L/2, P forward, and the reverse backward
        if isforward(direction)
            longitudinal_input!(u, p, k0, direction)
            longitudinal_sheet!(p, ε, cL / 2, direction)
            longitudinal_output_kick_input!(u, p, direction)
            kick_sheet!(p, ε, cK, direction)
            kick_output_longitudinal_input!(u, p, k0, direction)
            longitudinal_sheet!(p, ε, cL / 2, direction)
            longitudinal_output_modal_step!(u, p, slice, direction)
        else
            modal_step_longitudinal_input!(u, p, slice, k0, direction)
            longitudinal_sheet!(p, ε, cL / 2, direction)
            longitudinal_output_kick_input!(u, p, direction)
            kick_sheet!(p, ε, cK, direction)
            kick_output_longitudinal_input!(u, p, k0, direction)
            longitudinal_sheet!(p, ε, cL / 2, direction)
            longitudinal_output!(u, p, direction)
        end
    elseif isforward(direction)
        # K, L, P
        kick_input!(u, p, direction)
        kick_sheet!(p, ε, cK, direction)
        kick_output_longitudinal_input!(u, p, k0, direction)
        longitudinal_sheet!(p, ε, cL, direction)
        longitudinal_output_modal_step!(u, p, slice, direction)
    else
        # P, L, K
        modal_step_longitudinal_input!(u, p, slice, k0, direction)
        longitudinal_sheet!(p, ε, cL, direction)
        longitudinal_output_kick_input!(u, p, direction)
        kick_sheet!(p, ε, cK, direction)
        kick_output!(u, p, direction)
    end
    u
end

function propagate!(u::VectorialField, state, ::Nothing, p::VectorialBPM,
                    direction::Direction)
    T = real(eltype(u.Ex))
    k0 = T(2π / u.lambda)
    # Sheet coefficients: nrm_f normalizes the FFT round trip inside each sheet, and
    # sign(direction) turns each sheet into its inverse
    cK = Complex{T}(im * sign(direction) * p.dz * k0 * p.nrm_f)
    cL = Complex{T}(-im * sign(direction) * p.dz * p.nrm_f / p.eps_ref)
    # The forward pass reads the backward amplitudes of the state and keeps its own in
    # forward_amps; the backward pass reads these and writes the new state
    opposite, departing = isforward(direction) ? (state_arrays(state), p.forward_amps) :
        (p.forward_amps, state_arrays(state))
    # Interaction picture: phase factor of the current slice, updated after each slice
    P = isnothing(p.phasors) ? Unity() : p.phasors.current
    isnothing(p.phasors) || copyto!(P, isforward(direction) ? p.phasors.first_fwd :
                                                              p.phasors.first_bwd)
    nz = size(p.eps_xyz, 3)
    storage = isforward(direction) ? p.storage.fwd : p.storage.bwd
    # strang: half modal step 0 before the first slice (forward) or after it (backward)
    function half_step!()
        with_slice(storage, 1, opposite, departing, P) do slice
            modal_step!(u, p, merge(slice, (; factors = p.half_factors)), direction)
        end
    end
    p.strang && isforward(direction) && half_step!()
    for k in reverse(1:nz, direction)
        ε = slice_at(p.eps_xyz, k)
        factors = p.strang && k == nz ? p.half_factors : p.factors
        with_slice(storage, k + p.strang, opposite, departing, P) do slice
            propagate_slice!(u, p, ε, merge(slice, (; factors)), cK, cL, k0, direction)
        end
        isnothing(p.phasors) || (P .*= p.phasors.step)
    end
    p.strang && !isforward(direction) && half_step!()
    u
end
