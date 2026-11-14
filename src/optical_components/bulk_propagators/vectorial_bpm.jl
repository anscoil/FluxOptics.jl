struct VectorialBPM{M, T, N, R, P, E, F, K, B, S, A} <: AbstractBidirectionalComponent{M}
    trainability::Val{M}
    eps_xyz::N
    eps_ref::R
    dz::T
    p_f::P
    medium::E
    factors::F
    kx::K
    ky::K
    buffers::B
    storage::S         # stored slices and interpolation weights
    forward_amps::A    # forward amplitudes of the stored slices, for the backward pass
    slice_amps::B      # opposite, then departing amplitudes of the current slice
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
# Device: the interpolation runs through launch_combine!, on CPU and GPU alike.
# ---------------------------------------------------------------------------------------

struct SliceStorage{T}
    slots::Vector{Int}                # slot of each slice in the state, 0 if not stored
    nodes::Vector{NTuple{4, Int}}     # slots of the interpolation nodes of each slice
    weights::Vector{NTuple{4, T}}     # their cubic Lagrange weights
end

function SliceStorage(::Type{T}, nz::Integer, step::Integer) where {T}
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
        z = ntuple(l -> zs[j + l - 1], 4)
        weights[k] = ntuple(4) do l
            T(prod((k - z[m]) / (z[l] - z[m]) for m in 1:4 if m != l))
        end
    end
    SliceStorage(slots, nodes, weights)
end

n_stored(s::SliceStorage) = maximum(s.slots)

# Cubic interpolation of both opposite amplitudes of a slice: ψ_in holds the four node
# slices of amp1, then those of amp2
struct SliceInterpolation{T}
    w::NTuple{4, T}
end

function (op::SliceInterpolation)(::Direction, ψ, ψ_in)
    w1, w2, w3, w4 = op.w
    a1, b1, c1, d1, a2, b2, c2, d2 = ψ_in
    (w1 * a1 + w2 * b1 + w3 * c1 + w4 * d1, w1 * a2 + w2 * b2 + w3 * c2 + w4 * d2)
end

# Copy of both opposite amplitudes of a stored slice
copy_slice(::Direction, ψ, ψ_in) = ψ_in

# Opposite amplitudes of slice k, from the arrays `from` of the stored slices: a copy if
# k is stored, the cubic interpolation of its four nodes otherwise
function interpolate_slice!(amps::Tuple, from::Tuple, s::SliceStorage, k::Integer,
                            direction::Direction)
    j = s.slots[k]
    if j > 0
        launch_combine!(copy_slice, amps, map(b -> slice_at(b, j), from), (), direction)
    else
        n = s.nodes[k]
        a, b = from
        nodes = (map(i -> slice_at(a, i), n)..., map(i -> slice_at(b, i), n)...)
        launch_combine!(SliceInterpolation(s.weights[k]), amps, nodes, (), direction)
    end
    amps
end

# Departing amplitudes of slice k, kept in the arrays `to` if k is stored
function store_slice!(to::Tuple, amps::Tuple, s::SliceStorage, k::Integer)
    j = s.slots[k]
    j > 0 && foreach((b, a) -> copyto!(slice_at(b, j), a), to, amps)
    to
end

"""
    VectorialBPM(u, thickness, eps_xyz, eps_ref; step = 1, trainable = false)

Bidirectional vectorial split-step propagation through a layer of given thickness.
`eps_xyz` has shape (Nx, Ny, nz), one slice per z step, with Number (isotropic) or
ZDecEpsilon (z-decoupled tensor) entries. `eps_ref` is the scalar permittivity of the
reference medium in which the modal propagation is exact. The state stores slices spread
evenly from the first to the last, at most `step` slices apart, and interpolates the
others (step = 1 stores every slice).
"""
function VectorialBPM(u::VectorialField, thickness::Real,
                      eps_xyz::AbstractArray{<:Permittivity, 3}, eps_ref::Number;
                      step::Integer = 1, trainable::Bool = false)
    eltype(eps_xyz) <: Epsilon && throw(ArgumentError(
        "VectorialBPM handles isotropic and z-decoupled permittivities (ZDecEpsilon); " *
            "general tensors couple E⊥ and Ez and need the δ-variable scheme"))
    T = real(eltype(u.Ex))
    ns = size(u.Ex)[1:2]
    @assert size(eps_xyz)[1:2] == ns
    nz = size(eps_xyz, 3)
    dz = T(thickness / nz)
    # Conversion on the host first, so that the copy to the device never mixes types
    E = map_eltype(eltype(eps_xyz), T)
    eps_buf = adapt(get_backend(u.Ex), struct_of_arrays(collect(E, eps_xyz)))
    ε_r = isreal(eps_ref) ? T(real(eps_ref)) : Complex{T}(eps_ref)
    p_f, _ = make_fft_plans(similar(u.Ex), (1, 2); normalize = false)
    medium = VectorialMediumModes(u, ε_r)
    factors = propagation_factors.(medium.modes, dz)
    fx = collect(T, T(2π) .* fftfreq(ns[1], 1 / u.ds[1]))
    fy = collect(T, T(2π) .* fftfreq(ns[2], 1 / u.ds[2]))
    kx, ky = similar(u.Ex, T, ns), similar(u.Ex, T, ns)
    copyto!(kx, repeat(fx, 1, ns[2]))
    copyto!(ky, repeat(permutedims(fy), ns[1], 1))
    buffers = (similar(u.Ex), similar(u.Ex))
    storage = SliceStorage(T, nz, step)
    dims = (size(u.Ex)..., n_stored(storage))
    forward_amps = (similar(u.Ex, dims), similar(u.Ex, dims))
    slice_amps = (similar(u.Ex), similar(u.Ex))
    M = trainable ? Trainable : Static
    VectorialBPM(Val(M), eps_buf, ε_r, dz, p_f, medium, factors, kx, ky, buffers, storage,
                 forward_amps, slice_amps, T(1 / prod(ns)))
end

reference_medium(p::VectorialBPM) = p.medium

function alloc_fp_state(u::VectorialField, p::VectorialBPM)
    dims = (size(u.Ex)..., n_stored(p.storage))
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
# Both sheets follow the same pattern:
#   Input (Fourier) → IFFT → Product (real space) → FFT → Output (Fourier)
# The kick uses both buffers (bx, by), the longitudinal sheet uses bx only.
# ---------------------------------------------------------------------------------------

# Kick product, real space: (bx, by) holds E⊥ on input and ΔH⊥ on output,
# ΔHx = -c·ΔDy and ΔHy = c·ΔDx with ΔD = (ε⊥ - ε_r)·E⊥
struct KickProduct{C, R}
    c::C
    r::R
end

function (op::KickProduct)(ε, ::Direction, ψ, ψ_in)
    Ex, Ey = ψ
    Δ = ε - op.r
    (-op.c * Δ * Ey, op.c * Δ * Ex)
end

function (op::KickProduct)(εxx, εxy, εyx, εyy, ::Direction, ψ, ψ_in)
    Ex, Ey = ψ
    Dx, Dy = εxx * Ex + εxy * Ey, εyx * Ex + εyy * Ey
    (-op.c * (Dy - op.r * Ey), op.c * (Dx - op.r * Ex))
end

# Kick output, Fourier: H⊥ += ΔH⊥
function kick_output(::Direction, ψ, ψ_in)
    Hx, Hy = ψ
    ΔHx, ΔHy = ψ_in
    (Hx + ΔHx, Hy + ΔHy)
end

# Longitudinal input, Fourier: bx = D̂z
struct LongitudinalInput{T}
    k0::T
end

function (op::LongitudinalInput)(kx, ky, ::Direction, ψ, ψ_in)
    Hx, Hy = ψ_in
    (-(kx * Hy - ky * Hx) / op.k0,)
end

# Longitudinal product, real space: bx = c·(Δεzz/εzz)·D̂z, i.e. c·Δεzz·Ez
struct LongitudinalProduct{C, R}
    c::C
    r::R
end

function (op::LongitudinalProduct)(εzz, ::Direction, ψ, ψ_in)
    δ, = ψ
    (op.c * δ * (εzz - op.r) / εzz,)
end

# Longitudinal output, Fourier: E⊥ += k⊥·bx
function longitudinal_output(kx, ky, ::Direction, ψ, ψ_in)
    Ex, Ey = ψ
    δ, = ψ_in
    (Ex + kx * δ, Ey + ky * δ)
end

# ---------------------------------------------------------------------------------------
# The three operators
# ---------------------------------------------------------------------------------------

# K: 2 copies, 2 launches, 4 FFTs. ε is the permittivity slice.
function kick!(u::VectorialField, p::VectorialBPM, ε, cK, direction::Direction)
    bx, by = p.buffers
    copyto!(bx, u.Ex)
    copyto!(by, u.Ey)
    p.p_f.ift * bx
    p.p_f.ift * by
    launch_combine!(KickProduct(cK, p.eps_ref), (bx, by), (), kick_coefs(ε), direction)
    p.p_f.ft * bx
    p.p_f.ft * by
    launch_combine!(kick_output, (u.Hx, u.Hy), (bx, by), (), direction)
    u
end

# L: 3 launches, 2 FFTs. ε is the permittivity slice, of which only εzz is read.
function longitudinal!(u::VectorialField, p::VectorialBPM, ε, cL, k0,
                       direction::Direction)
    δ = first(p.buffers)
    k_grid = (p.kx, p.ky)
    launch_combine!(LongitudinalInput(k0), (δ,), (u.Hx, u.Hy), k_grid, direction)
    p.p_f.ift * δ
    launch_combine!(LongitudinalProduct(cL, p.eps_ref), (δ,), (), (zz_component(ε),),
                    direction)
    p.p_f.ft * δ
    launch_combine!(longitudinal_output, (u.Ex, u.Ey), (δ,), k_grid, direction)
    u
end

# P: 1 launch, stored is the state of the slice
function free_propagation!(u::VectorialField, stored::Tuple, p::VectorialBPM,
                           direction::Direction)
    launch_modal!(propagate_modes, (u.Ex, u.Ey, u.Hx, u.Hy), stored,
                  (p.medium.modes, p.factors), direction)
    u
end

# ---------------------------------------------------------------------------------------
# Slice and slice loop
# ---------------------------------------------------------------------------------------

function propagate_slice!(u::VectorialField, stored::Tuple, p::VectorialBPM, ε,
                          cK, cL, k0, direction::Direction)
    if isforward(direction)
        kick!(u, p, ε, cK, direction)
        longitudinal!(u, p, ε, cL, k0, direction)
        free_propagation!(u, stored, p, direction)
    else
        free_propagation!(u, stored, p, direction)
        longitudinal!(u, p, ε, cL, k0, direction)
        kick!(u, p, ε, cK, direction)
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
    for k in reverse(1:size(p.eps_xyz, 3), direction)
        ε = slice_at(p.eps_xyz, k)
        interpolate_slice!(p.slice_amps, opposite, p.storage, k, direction)
        propagate_slice!(u, p.slice_amps, p, ε, cK, cL, k0, direction)
        store_slice!(departing, p.slice_amps, p.storage, k)
    end
    u
end
