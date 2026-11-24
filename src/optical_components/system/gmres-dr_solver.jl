# ---------------------------------------------------------------------------------------
# GMRES-DR
#
# GMRES with deflated restarting (Morgan, SIAM J. Sci. Comput. 24, 20–37, 2002). Each
# cycle builds a Krylov basis of at most `memory` vectors. At the restart, the
# `deflation` harmonic Ritz vectors of smallest harmonic Ritz values ν (approximations of
# the eigenvalues of A closest to 0, the modes that slow GMRES down) are kept together
# with the residual: the next cycle starts from the subspace they span instead of from
# the residual alone, so that the slow modes are not lost at each restart, unlike in
# restarted GMRES. The Arnoldi relation A·V[:, 1:k] = V[:, 1:k+1]·H[1:k+1, 1:k] carries
# over to the kept vectors, with a dense leading block of H, so that the residual stays
# minimized over the whole augmented subspace. Cycles after the first one cost
# memory - deflation products by the operator.
#
# Batches. The workspace solves a batch of B independent problems sharing one vector (see
# Batched windows in the shared tools): blocks of P entries, block c to element
# c mod B + 1. A single problem is the case P = n, B = 1. The operator is applied to the
# whole batch at once, but each element runs its own GMRES-DR: its own basis (the rows of
# the element in the columns of C), projected problem, cycles, restarts, residual checks
# and convergence. At each product, each element gives its own input: the column of its
# basis to expand (Arnoldi), its iterate x (true residual check), or 0 once converged
# (frozen). Each element thus makes exactly the iterations it would make alone, and the
# batch costs the products of its slowest element. The tolerance of each element is
# max(atol/√B, rtol·‖b‖) on its rows. GmresDRSolver takes the batch from the dimensions
# of the fields after the first two.
#
# Orthogonalization. The Arnoldi process runs on I - A, which spans the same Krylov
# subspace: (I - A)·v = v - A·v, then H_A = [I; 0] - H_{I-A}. For a fixed-point operator
# A = I - G, A·v is mostly v itself, already in the basis, and the first pass of classical
# Gram–Schmidt would remove most of its norm, which signals a loss of orthogonality;
# (I - A)·v = G·v has no such artificial cancellation. The method depends on the basis:
# - host (Array): one pass of modified Gram–Schmidt, as in Krylov.jl, whose loss of
#   orthogonality does not harm GMRES (Paige, Rozložník, Strakoš, SIAM J. Matrix Anal.
#   Appl. 28, 2006). The part of w of an element stays in cache while each basis vector
#   is read once from memory, against four reads with two passes of classical
#   Gram–Schmidt;
# - device: modified Gram–Schmidt would need one reduction per basis vector, so one pass
#   of classical Gram–Schmidt, repeated only if the norm drops below 1/√2 of its norm
#   before the pass (Daniel, Gragg, Kaufman, Stewart): two reductions per iteration (the
#   dot products with the norm before, then the norm after), the basis read twice per
#   pass. The second passes are counted in reorthogonalizations. The cancellation can
#   also be genuine: for the round trip of a VectorialBPM, G·V[:, j] lies almost
#   entirely in the span of the basis (median ratio of the norms after and before the
#   pass about 0.005), the second pass is then made at almost every iteration, and a
#   single pass of classical Gram–Schmidt loses orthogonality completely.
#
# The residual of each projected problem is updated by Givens rotations, after a QR
# factorization of the dense leading block at each restart. The small
# (memory + 1) × memory projected problems live on the host. The restart
# V[:, 1:k+1] ← V·P is done in place, by blocks of at most chunk rows of the element, or
# by one product V·p when no vector is kept.
#
# Basis storage. The basis is kept in C, of element type `storage`: the working precision
# by default, or a lower one (Complex{Float16}, or ComplexF32 for ComplexF64 work), which
# only changes how the vectors are kept in memory: every product with them is done in the
# working precision, as in compressed basis GMRES (Aliaga, Anzt et al., Int. J. High
# Perform. Comput. Appl. 37, 2023). The basis vectors of an element have unit norm and are
# stored scaled by s0 = √(n/B) (s0 = 1 in the working precision): their root mean square
# is then 1 and their modulus at most s0, well within the range of Float16 (normal
# numbers from 6e-5 to 65504), so that only entries far below the rms lose relative
# precision. The stored basis is not exactly the computed one (relative rounding about
# 5e-4 in Float16): the estimated residual can then drift from the true one, and
# check_residual restarts from the true residual when they disagree, as an iterative
# refinement.
#
# Resuming. A solve stopped by maxit can be continued on the same operator and right-hand
# side (resume = true): the cycles in progress go on from their basis, projected problem
# and kept vectors, so that the products are exactly those of a single solve. When it
# stops in the middle of a cycle, the solve still returns the minimizer over the basis
# built so far, x + V·y, without ending the cycle: the iterate at the start of the cycle
# is kept in v, and a resumed solve restores it. The tolerances are those of the first
# solve.
#
# Memory: memory + 1 stored vectors, 3 vectors in the working precision (x, w, v), the
# projected problems of the B elements on the host, and min(chunk, P) × (memory +
# deflation + 2) entries for the restarts with kept vectors (none when deflation = 0).
# Only the columns of the basis that a solve reaches are touched, but the whole basis is
# allocated with the workspace: build it once and reuse it for several solves of the same
# size, its allocation at every solve costing about as much in garbage collection as the
# iterations for a large memory.
# ---------------------------------------------------------------------------------------

const FROZEN = Int8(0)      # converged, or stopped: input 0
const ARNOLDI = Int8(1)     # in a cycle: input V[:, j + 1]
const CHECK = Int8(2)       # true residual check: input x

"""
    GmresDRWorkspace(m, n, S; memory = 40, deflation = 20, chunk = 2^12,
                     check_residual = true, storage = nothing, layout = nothing,
                     n_tasks = 1)

- memory: maximum dimension of the Krylov basis, i.e. restart length;
- deflation: number of harmonic Ritz vectors kept at each restart (0: restarted GMRES;
  memory larger than the number of products: full GMRES);
- chunk: number of rows of the basis processed at once at a restart (buffer size);
- check_residual: when the estimated residual of an element reaches its tolerance,
  compute its true residual b - A x (one more product); if it is still too large,
  restart from it;
- storage: element type of the stored basis, e.g. Complex{Float16}; nothing keeps the
  working precision;
- layout: BatchLayout(P, B) for a batch of B problems, nothing for a single one;
- n_tasks: Julia tasks on the host: the orthogonalization shares the elements among
  them, the other products with the basis their rows (BLAS threads for a single problem
  in the working precision).

After a solve: `element_niter[e]` products made for element e (stats.niter counts the
products of the batch), `converged[e]`, the residual estimates of each element in
`residuals[e]`, its restarts from the true residual (estimate converged, true residual
not) in `refinements[e]`, and its second Gram–Schmidt passes on a device in
`reorthogonalizations[e]`. A solve stopped by maxit can be continued with
`gmresdr!(ws, op, b; resume = true, ...)` (see Resuming above).
"""
struct GmresDRWorkspace{T, R, S, M, W, D}
    layout::BatchLayout
    x::S                          # iterate, then solution
    w::S                          # operator outputs: new Arnoldi vectors, true residuals
    v::S                          # operator inputs
    C::M                          # stored Krylov bases s0·V, n × (memory + 1)
    s0::R                         # storage scale
    buffers::W                    # buffers of the products with C
    Pd::D                         # restart matrix of an element on the device
    chunk_in::D                   # rows of V, at most chunk × (memory + 1)
    chunk_out::D                  # rows of V·P, at most chunk × (deflation + 1)
    # Projected problems, last index: element
    H::Array{T, 3}                # A·V[:, 1:j] = V[:, 1:j+1]·H[1:j+1, 1:j]
    Rq::Array{T, 3}               # H reduced to upper triangular form
    c::Matrix{T}                  # Vᴴ·r at the start of the cycle
    g::Matrix{T}                  # c rotated like Rq: |g[j+1]| is the residual norm
    y::Matrix{T}                  # weights of the basis vectors, x += V·y
    s::Matrix{T}                  # residual c - H·y of the projected problem
    Q0::Array{T, 3}               # Q of the leading block after a restart
    P::Array{T, 3}                # restart matrix, (memory + 1) × (deflation + 1)
    rot::Matrix{LinearAlgebra.Givens{T}}
    hh::Matrix{T}                 # host coefficients, (memory + 2) × B
    t::Vector{T}                  # host buffer, deflation + 1
    # State of the elements
    phase::Vector{Int8}           # FROZEN, ARNOLDI or CHECK
    j::Vector{Int}                # columns of H formed in the current cycle
    k0::Vector{Int}               # vectors kept at the start of the current cycle
    tol::Vector{R}
    estimate::Vector{R}           # residual estimate of the last Arnoldi step
    before::Vector{R}             # norm before the Gram–Schmidt pass
    breakdown::Vector{Bool}
    ending::Vector{Bool}          # cycle ending at this step
    converged::Vector{Bool}
    mask::Vector{Bool}            # elements of a batched norm
    again::Vector{Bool}           # elements needing a second Gram–Schmidt pass
    kept::Vector{Int}             # vectors kept by the restarting elements, -1 otherwise
    norms::Vector{R}              # norms of the elements (host buffers)
    after::Vector{R}
    rnorm::Vector{R}
    element_niter::Vector{Int}
    residuals::Vector{Vector{R}}
    refinements::Vector{Int}
    reorthogonalizations::Vector{Int}
    # Control vectors of the batched products (host)
    ncols::Vector{Int32}
    cols::Vector{Int32}
    modes::Vector{Int32}
    scales::Vector{R}
    memory::Int
    deflation::Int
    chunk::Int
    check_residual::Bool
    n_tasks::Int
    stats::IterativeStats
    pending::Vector{Bool}         # x includes the partial cycle of the element (resume)
end

function GmresDRWorkspace(m, n, S::Type; memory = 40, deflation = 20, chunk = 2^12,
                          check_residual = true, storage = nothing, layout = nothing,
                          n_tasks = 1)
    0 ≤ deflation < memory || throw(ArgumentError("0 ≤ deflation < memory required"))
    L = isnothing(layout) ? BatchLayout(n, 1) : layout
    n % (L.P * L.B) == 0 ||
        throw(ArgumentError("the length $n is not a multiple of P·B = $(L.P * L.B)"))
    B = L.B
    x = S(undef, n)
    T = eltype(x)
    R = real(T)
    T <: Complex || throw(ArgumentError("complex element type required"))
    Tc = isnothing(storage) ? T : storage
    Tc <: Complex || throw(ArgumentError("complex storage type required"))
    s0 = Tc == T ? one(R) : sqrt(R(n ÷ B))
    C = similar(x, Tc, n, memory + 1)
    # Buffers of the restart with kept vectors only (k = 0 needs one product V·p)
    k1, rows = deflation + 1, deflation > 0 ? min(chunk, L.P) : 0
    M1 = memory + 1
    GmresDRWorkspace(L, x, similar(x), similar(x), C, s0,
                     BatchBuffers(x, L, M1, n_tasks), similar(x, M1, k1),
                     similar(x, rows, M1), similar(x, rows, k1),
                     zeros(T, M1, memory, B), zeros(T, M1, memory, B), zeros(T, M1, B),
                     zeros(T, M1, B), zeros(T, memory, B), zeros(T, M1, B),
                     zeros(T, k1, k1, B), zeros(T, M1, k1, B),
                     Matrix{LinearAlgebra.Givens{T}}(undef, memory, B),
                     zeros(T, memory + 2, B), zeros(T, k1),
                     fill(FROZEN, B), zeros(Int, B), zeros(Int, B), zeros(R, B),
                     zeros(R, B), zeros(R, B), fill(false, B), fill(false, B),
                     fill(false, B), fill(false, B), fill(false, B), fill(-1, B),
                     zeros(R, B), zeros(R, B), zeros(R, B),
                     zeros(Int, B), [R[] for _ in 1:B], zeros(Int, B), zeros(Int, B),
                     zeros(Int32, B), zeros(Int32, B), zeros(Int32, B), zeros(R, B),
                     memory, deflation, chunk, check_residual, n_tasks, IterativeStats(),
                     fill(false, B))
end

Krylov.solution(ws::GmresDRWorkspace) = ws.x

"""
    GmresDRSolver(system; memory = 40, deflation = 20, chunk = 2^12,
                  check_residual = true, storage = nothing, n_tasks = 1)

GMRES-DR for the fixed point of the system (see GmresDRWorkspace). The dimensions of the
fields after the first two are batch dimensions: their elements are solved
independently, each with its own basis, iterations and convergence, and every array of
the state must then start with the dimensions of the fields. Without batch dimension,
the state is a single problem, whatever its arrays.
"""
function GmresDRSolver(system::BidirectionalSystem; memory = 40, deflation = 20,
                       chunk = 2^12, check_residual = true, storage = nothing,
                       n_tasks = 1)
    FixedPointSolver(system, GmresDRWorkspace; memory, deflation, chunk, check_residual,
                     storage, n_tasks, layout = batch_layout(system))
end

state_leaves(::Nothing) = ()
state_leaves(a::AbstractArray) = (a,)
function state_leaves(t::Union{Tuple, NamedTuple})
    mapreduce(state_leaves, (a, b) -> (a..., b...), values(t); init = ())
end

# Blocks of the fields (P = Nx·Ny) and batch size B, checked against the state; nothing
# (single problem) without batch dimension
function batch_layout(s::BidirectionalSystem)
    dims = size(s.s_in)
    length(dims) ≥ 2 || throw(ArgumentError("fields of at least two dimensions required"))
    B = prod(dims[3:end]; init = 1)
    B == 1 && return nothing
    leaves = state_leaves(ca_views(s.fp_state, ComponentArrays.valkeys(s.fp_state)))
    for a in leaves
        ndims(a) ≥ length(dims) && size(a)[1:length(dims)] == dims ||
            throw(ArgumentError("state array of size $(size(a)) does not start with " *
                                "the dimensions $dims of the fields"))
    end
    BatchLayout(dims[1] * dims[2], B)
end

# ---------------------------------------------------------------------------------------
# Products with the stored bases
# ---------------------------------------------------------------------------------------

function element_dots!(ws::GmresDRWorkspace, w)
    element_dots!(ws.hh, ws.buffers, ws.C, w, ws.ncols, ws.layout, ws.n_tasks)
end

function element_combine!(ws::GmresDRWorkspace, target)
    element_combine!(target, ws.buffers, ws.C, ws.hh, ws.ncols, ws.layout, ws.n_tasks)
end

function gather!(ws::GmresDRWorkspace)
    gather!(ws.v, ws.buffers, ws.C, ws.x, ws.cols, ws.s0, ws.layout, ws.n_tasks)
end

function transform!(ws::GmresDRWorkspace, b)
    transform!(ws.w, ws.buffers, ws.v, b, ws.modes, ws.layout, ws.n_tasks)
end

function store!(ws::GmresDRWorkspace, w)
    store!(ws.C, ws.buffers, w, ws.cols, ws.scales, ws.layout, ws.n_tasks)
end

# Norms of w on the rows of the elements with mask[e] into out (others: 0, skipped)
function element_norms!(out, ws::GmresDRWorkspace, w, mask)
    ws.ncols .= ifelse.(mask, Int32(0), Int32(-1))
    element_dots!(ws, w)
    for e in 1:ws.layout.B
        out[e] = mask[e] ? sqrt(real(ws.hh[1, e])) : zero(ws.s0)
    end
    out
end

# One pass of classical Gram–Schmidt for the elements with ncols[e] = j > 0: w minus its
# components on V[:, 1:j], added to H[1:j, j, e]; ws.before[e] receives the norm of w
# before the pass
function cgs_pass!(ws::GmresDRWorkspace, w)
    s0, hh = ws.s0, ws.hh
    element_dots!(ws, w)                        # hh = s0·Vᴴ·w, then ‖w‖²
    for e in 1:ws.layout.B
        j = Int(ws.ncols[e])
        j > 0 || continue
        ws.before[e] = sqrt(real(hh[j + 1, e]))
        for i in 1:j
            ws.H[i, j, e] += hh[i, e] / s0
            hh[i, e] *= -inv(s0)^2              # w -= V·(Vᴴ·w) = C·(-s0·Vᴴ·w / s0²)
        end
    end
    element_combine!(ws, w)
end

# Orthogonalization of w for the elements with ncols[e] = j ≥ 0 (see Orthogonalization):
# its components on V[:, 1:j] added to H[1:j, j, e], ws.after[e] its norm after (j = 0:
# its norm). Host: one pass of modified Gram–Schmidt.
function gram_schmidt!(ws::GmresDRWorkspace{<:Any, <:Any, <:Any, <:Array}, w)
    s0, hh, after = ws.s0, ws.hh, ws.after
    element_mgs!(hh, ws.C, w, ws.ncols, inv(s0)^2, ws.layout, ws.n_tasks)
    for e in 1:ws.layout.B
        j = Int(ws.ncols[e])
        j < 0 && continue
        for i in 1:j
            ws.H[i, j, e] += hh[i, e] / s0
        end
        after[e] = sqrt(real(hh[j + 1, e]))
    end
    after
end

# Device: classical Gram–Schmidt, repeated where the norm dropped below 1/√2 of its norm
# before the pass
function gram_schmidt!(ws::GmresDRWorkspace, w)
    B, after = ws.layout.B, ws.after
    arnoldi, again = ws.mask, ws.again
    arnoldi .= ws.ncols .> 0
    cgs_pass!(ws, w)
    for e in 1:B
        ws.ncols[e] == 0 && (after[e] = sqrt(real(ws.hh[1, e])))
    end
    element_norms!(ws.norms, ws, w, arnoldi)
    after .= ifelse.(arnoldi, ws.norms, after)
    again .= arnoldi .& (after .< ws.before ./ sqrt(real(eltype(ws.H))(2)))
    if any(again)
        for e in 1:B
            ws.ncols[e] = again[e] ? ws.j[e] + 1 : -1
        end
        ws.reorthogonalizations .+= again
        cgs_pass!(ws, w)
        element_norms!(ws.norms, ws, w, again)
        after .= ifelse.(again, ws.norms, after)
    end
    after
end

# ---------------------------------------------------------------------------------------
# Projected problems, one element at a time
# ---------------------------------------------------------------------------------------

# Start of a cycle of element e with k0 kept vectors: QR factorization of the dense
# leading block H[1:k0+1, 1:k0] (Q0), applied to c. Without kept vector, g = c.
function start_cycle!(ws::GmresDRWorkspace, e, k0)
    T = eltype(ws.H)
    ws.k0[e], ws.j[e] = k0, k0
    fill!(view(ws.Rq, :, :, e), zero(T))
    copyto!(view(ws.g, :, e), view(ws.c, :, e))
    k0 == 0 && return
    a = 1:(k0 + 1)
    lead, Q0, t = view(ws.H, a, 1:k0, e), view(ws.Q0, a, a, e), view(ws.t, a)
    fill!(Q0, zero(T))
    for i in a
        Q0[i, i] = one(T)
    end
    lmul!(qr(lead).Q, Q0)                       # small, on the host
    mul!(view(ws.Rq, a, 1:k0, e), Q0', lead)
    mul!(t, Q0', view(ws.c, a, e))
    copyto!(view(ws.g, a, e), t)
    nothing
end

# Fresh cycle of element e from its residual of norm r, its first basis vector being
# stored at the same step (cols, scales)
function fresh_cycle!(ws::GmresDRWorkspace, e, r)
    T = eltype(ws.H)
    fill!(view(ws.H, :, :, e), zero(T))
    fill!(view(ws.c, :, e), zero(T))
    ws.c[1, e] = r
    ws.phase[e] = ARNOLDI
    ws.cols[e], ws.scales[e] = 1, ws.s0 / r
    start_cycle!(ws, e, 0)
end

# Column j of H of element e brought to upper triangular form in Rq: Q0ᴴ on the rows of
# the leading block, the rotations of the previous Arnoldi columns, then a new rotation
# that zeroes the subdiagonal entry, also applied to g. Returns the residual estimate.
function reduce_column!(ws::GmresDRWorkspace, e, j)
    k0 = ws.k0[e]
    col = view(ws.Rq, 1:(j + 1), j, e)
    copyto!(col, view(ws.H, 1:(j + 1), j, e))
    if k0 > 0
        a = 1:(k0 + 1)
        lead, t = view(col, a), view(ws.t, a)
        mul!(t, view(ws.Q0, a, a, e)', lead)
        copyto!(lead, t)
    end
    for i in (k0 + 1):(j - 1)
        lmul!(ws.rot[i, e], col)
    end
    G, r = givens(col[j], col[j + 1], j, j + 1)
    col[j], col[j + 1] = r, zero(r)
    ws.rot[j, e] = G
    lmul!(G, view(ws.g, 1:(j + 1), e))
    abs(ws.g[j + 1, e])
end

# End of a cycle of j columns of element e: weights y = Rq⁻¹·g, set as the coefficients
# of x += V·y (hh, ncols), and residual of the projected problem s = c - H·y, i.e.
# b - A x = V[:, 1:j+1]·s
function end_cycle!(ws::GmresDRWorkspace, e)
    j = ws.j[e]
    y = view(ws.y, 1:j, e)
    copyto!(y, view(ws.g, 1:j, e))
    ldiv!(UpperTriangular(view(ws.Rq, 1:j, 1:j, e)), y)
    view(ws.hh, 1:j, e) .= y ./ ws.s0
    ws.ncols[e] = j
    s = view(ws.s, 1:(j + 1), e)
    copyto!(s, view(ws.c, 1:(j + 1), e))
    mul!(s, view(ws.H, 1:(j + 1), 1:j, e), y, -1, true)
    nothing
end

# p ← p minus its components on the columns of Q (orthonormal), twice
function orthogonalize!(p, Q)
    for pass in 1:2, i in axes(Q, 2)
        q = view(Q, :, i)
        p .-= dot(q, p) .* q
    end
    p
end

# Restart of element e after a full cycle: P = [harmonic Ritz vectors (padded with a
# zero), residual s], orthonormalized; new H = Pᴴ·H·P[1:memory, 1:k], new c = Pᴴ·s.
# Harmonic Ritz vectors too close to the span of the previous ones are skipped. Returns
# the number k of vectors kept (≤ deflation); the basis is rotated afterwards.
function deflate!(ws::GmresDRWorkspace, e)
    M = ws.memory
    H, P = view(ws.H, :, :, e), view(ws.P, :, :, e)
    T = eltype(H)
    fill!(P, zero(T))
    k = 0
    if ws.deflation > 0
        # Harmonic Ritz pairs: (Hm + |h|²·Hm⁻ᴴ·e_M·e_Mᵀ)·u = ν·u, h = H[M+1, M]
        Hm = H[1:M, 1:M]
        f = zeros(T, M)
        f[M] = one(T)
        f = lu(Hm') \ f
        Hm[:, M] .+= abs2(H[M + 1, M]) .* f
        E = eigen!(Hm)
        for i in sortperm(E.values; by = abs)
            k == ws.deflation && break
            p = view(P, 1:M, k + 1)
            copyto!(p, view(E.vectors, :, i))
            orthogonalize!(p, view(P, 1:M, 1:k))
            np = norm(p)
            if np > 1e-8
                p ./= np
                k += 1
            else
                fill!(p, zero(T))
            end
        end
    end
    s = view(ws.s, 1:(M + 1), e)
    p = view(P, :, k + 1)
    copyto!(p, s)
    orthogonalize!(p, view(P, :, 1:k))
    p ./= norm(p)
    Pk = view(P, :, 1:(k + 1))
    Hn = Pk' * (H * view(P, 1:M, 1:k))          # (k + 1) × k, small
    cn = Pk' * s
    fill!(H, zero(T))
    H[1:(k + 1), 1:k] .= Hn
    c = view(ws.c, :, e)
    fill!(c, zero(T))
    c[1:(k + 1)] .= cn
    k
end

# Rows of element e by ranges of at most `rows` entries
function element_row_ranges(L::BatchLayout, n, e, rows)
    ranges = UnitRange{Int}[]
    for c in (e - 1):L.B:(n ÷ L.P - 1), first in 0:rows:(L.P - 1)
        push!(ranges, (c * L.P + first + 1):(c * L.P + min(first + rows, L.P)))
    end
    ranges
end

# V[:, 1:k+1] ← V·P[:, 1:k+1] on the rows of element e (k ≥ 1), in place, by ranges of
# rows computed in the working precision
function rotate_element!(ws::GmresDRWorkspace, e, k)
    C, s0 = ws.C, ws.s0
    copyto!(ws.Pd, ws.P[:, :, e])               # host copy first: a device copy needs it
    Pk = view(ws.Pd, :, 1:(k + 1))
    for rows in element_row_ranges(ws.layout, size(C, 1), e, size(ws.chunk_in, 1))
        rows_in = view(ws.chunk_in, 1:length(rows), :)
        rows_out = view(ws.chunk_out, 1:length(rows), 1:(k + 1))
        rows_in .= view(C, rows, :) ./ s0
        mul!(rows_out, rows_in, Pk)
        view(C, rows, 1:(k + 1)) .= rows_out .* s0
    end
    nothing
end

# Restarts of the elements with kept[e] ≥ 0 vectors (-1: not restarting): bases rotated,
# then the last kept vector reorthogonalized against the others (rounding errors only)
# and normalized; with k = 0, V[:, 1] ← V·p by one product with the basis
function restart_bases!(ws::GmresDRWorkspace, kept)
    B, M1, s0 = ws.layout.B, ws.memory + 1, ws.s0
    for e in 1:B
        kept[e] > 0 && rotate_element!(ws, e, kept[e])
    end
    # k = 0: v = V·p, stored as V[:, 1]
    if any(==(0), kept)
        ws.cols .= -1
        gather!(ws)                             # v = 0
        for e in 1:B
            ws.ncols[e] = kept[e] == 0 ? M1 : 0
            kept[e] == 0 && (view(ws.hh, 1:M1, e) .= view(ws.P, :, 1, e) ./ s0)
        end
        element_combine!(ws, ws.v)
        ws.cols .= ifelse.(kept .== 0, Int32(1), Int32(0))
        ws.scales .= s0
        store!(ws, ws.v)
    end
    # k > 0: last kept vector reorthogonalized and normalized
    if any(>(0), kept)
        ws.cols .= ifelse.(kept .> 0, Int32.(kept .+ 1), Int32(-1))
        gather!(ws)
        ws.ncols .= ifelse.(kept .> 0, Int32.(kept), Int32(-1))
        element_dots!(ws, ws.v)                 # hh = s0·Vᴴ·v
        for e in 1:B
            k = kept[e]
            k > 0 && (view(ws.hh, 1:k, e) .*= -inv(s0)^2)
        end
        element_combine!(ws, ws.v)
        ws.mask .= kept .> 0
        nv = element_norms!(ws.norms, ws, ws.v, ws.mask)
        ws.cols .= ifelse.(kept .> 0, Int32.(kept .+ 1), Int32(0))
        ws.scales .= ifelse.(kept .> 0, s0 ./ nv, zero(s0))
        store!(ws, ws.v)
    end
    nothing
end

# ---------------------------------------------------------------------------------------
# Iterations
# ---------------------------------------------------------------------------------------

# Arnoldi steps of the elements in a cycle, on the outputs w = (I - A)·V[:, j] (j =
# ws.j[e] + 1): orthogonalized against V[:, 1:j], then H[1:j+1, j] = [e_j; 0] -
# [components of w; -‖w‖] and V[:, j+1] = -w/‖w‖ set to be stored. Returns the true
# residual norms of the elements in CHECK, measured by the same first reduction.
function arnoldi_steps!(ws::GmresDRWorkspace)
    B, H = ws.layout.B, ws.H
    T = eltype(H)
    rnorm = fill!(ws.rnorm, zero(ws.s0))
    for e in 1:B
        phase = ws.phase[e]
        if phase == ARNOLDI
            j = ws.j[e] + 1
            fill!(view(H, 1:(j + 1), j, e), zero(T))
            ws.ncols[e] = j
        else
            ws.ncols[e] = phase == CHECK ? 0 : -1
        end
    end
    after = gram_schmidt!(ws, ws.w)
    for e in 1:B
        if ws.phase[e] == ARNOLDI
            j = ws.j[e] + 1
            h = view(H, 1:j, j, e)
            h .= .-h
            h[j] += one(T)
            H[j + 1, j, e] = after[e]
            ws.breakdown[e] = iszero(after[e])
            # V[:, j+1] = -w/β, so that A·V[:, j] = V[:, 1:j+1]·H[1:j+1, j], H[j+1, j] = β
            ws.cols[e] = ws.breakdown[e] ? 0 : j + 1
            ws.scales[e] = ws.breakdown[e] ? zero(ws.s0) : -ws.s0 / after[e]
        else
            ws.phase[e] == CHECK && (rnorm[e] = after[e])
            ws.cols[e] = 0
        end
    end
    rnorm
end

"""
Solve A x = b from x = 0 by GMRES-DR; at most maxit products by A. With resume = true,
continue the previous solve of the same A and b (stopped by maxit) where it stopped,
with its tolerances (atol and rtol are then ignored), for at most maxit more products.
"""
function gmresdr!(ws::GmresDRWorkspace, op, b; atol, rtol, maxit, resume::Bool = false)
    L, x, w = ws.layout, ws.x, ws.w
    B, M = L.B, ws.memory
    T = eltype(x)
    R = real(T)
    if resume
        # Iterate at the start of the cycles in progress, if a partial cycle was added
        any(ws.pending) && copyto!(x, ws.v)
        fill!(ws.pending, false)
    else
        fill!(x, zero(T))
        fill!(ws.pending, false)
        fill!(ws.element_niter, 0)
        fill!(ws.refinements, 0)
        fill!(ws.reorthogonalizations, 0)
        fill!(ws.converged, false)
        # Residuals of x = 0, without a product: b itself
        copyto!(w, b)
        fill!(ws.mask, true)
        r0 = element_norms!(ws.norms, ws, w, ws.mask)
        ws.tol .= max.(R(atol) / sqrt(R(B)), R(rtol) .* r0)
        for e in 1:B
            empty!(ws.residuals[e])
            push!(ws.residuals[e], r0[e])
            if r0[e] ≤ ws.tol[e]
                ws.phase[e], ws.converged[e], ws.cols[e] = FROZEN, true, 0
            else
                fresh_cycle!(ws, e, r0[e])
            end
        end
        store!(ws, w)
    end
    niter = 0
    kept = ws.kept
    while any(!=(FROZEN), ws.phase) && niter < maxit
        # Inputs of the elements, one product for the batch
        for e in 1:B
            phase = ws.phase[e]
            ws.cols[e] = phase == ARNOLDI ? ws.j[e] + 1 : phase == CHECK ? 0 : -1
            ws.modes[e] = phase
            phase == FROZEN || (ws.element_niter[e] += 1)
        end
        gather!(ws)
        mul!(w, op, ws.v)
        niter += 1
        transform!(ws, b)                       # (I - A)·v, b - A·x, or 0
        rnorm = arnoldi_steps!(ws)
        # Arnoldi elements: new columns; CHECK elements: converged or fresh cycle
        fill!(ws.ending, false)
        for e in 1:B
            if ws.phase[e] == ARNOLDI
                j = ws.j[e] + 1
                ws.j[e] = j
                ws.estimate[e] = reduce_column!(ws, e, j)
                push!(ws.residuals[e], ws.estimate[e])
                ws.ending[e] = ws.estimate[e] ≤ ws.tol[e] || ws.breakdown[e] || j == M
            elseif ws.phase[e] == CHECK
                push!(ws.residuals[e], rnorm[e])
                if rnorm[e] ≤ ws.tol[e]
                    ws.phase[e], ws.converged[e] = FROZEN, true
                else
                    ws.refinements[e] += 1
                    fresh_cycle!(ws, e, rnorm[e])
                end
            end
        end
        store!(ws, w)
        any(ws.ending) || continue
        # Ends of cycles: x += V·y for the ending elements, then their next phase
        for e in 1:B
            ws.ncols[e] = -1
            ws.ending[e] && end_cycle!(ws, e)
        end
        element_combine!(ws, x)
        fill!(kept, -1)
        for e in 1:B
            ws.ending[e] || continue
            if ws.estimate[e] ≤ ws.tol[e] || ws.breakdown[e]
                if ws.check_residual
                    # True residual at the next product; a solve stopped before it
                    # reports the element unconverged, and a resumed one starts with it
                    ws.phase[e] = CHECK
                else
                    ws.phase[e] = FROZEN
                    ws.converged[e] = ws.estimate[e] ≤ ws.tol[e]
                end
            else                                # full cycle: deflated restart
                kept[e] = deflate!(ws, e)
            end
        end
        if any(≥(0), kept)
            restart_bases!(ws, kept)
            for e in 1:B
                kept[e] ≥ 0 && start_cycle!(ws, e, kept[e])
            end
        end
    end
    # Stopped in the middle of cycles: x + V·y for them, the iterate at the start of
    # the cycles kept in v for a resumed solve (the cycles themselves are not ended)
    for e in 1:B
        ws.pending[e] = ws.phase[e] == ARNOLDI && ws.j[e] > ws.k0[e]
    end
    if any(ws.pending)
        copyto!(ws.v, x)
        for e in 1:B
            ws.ncols[e] = -1
            ws.pending[e] && end_cycle!(ws, e)
        end
        element_combine!(ws, x)
    end
    ws.stats.niter = niter
    ws.stats.solved = all(ws.converged)
    ws.stats
end

supports_resume(::GmresDRWorkspace) = true

function krylov_solve!(solver::BidirectionalSolver{<:GmresDRWorkspace},
                       op::LinearOperator; atol = 0, rtol = default_rtol(solver),
                       itmax = 0, resume::Bool = false, kwargs...)
    gmresdr!(solver.workspace, op, getdata(solver.r); atol, rtol,
             maxit = max_products(op, itmax), resume)
end
