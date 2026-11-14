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
# Orthogonalization: classical Gram–Schmidt with one reorthogonalization (two passes per
# iteration). The residual of the projected problem is updated by Givens rotations, after
# a QR factorization of the dense leading block at each restart. The small
# (memory + 1) × memory projected problem lives on the host. The restart
# V[:, 1:k+1] ← V·P is done in place, by blocks of chunk rows.
#
# Basis storage. The basis is kept in C, of element type `storage`: the working precision
# by default, or a lower one (Complex{Float16}, or ComplexF32 for ComplexF64 work), which
# only changes how the vectors are kept in memory: every product with them is done in the
# working precision, as in compressed basis GMRES (Aliaga, Anzt et al., Int. J. High
# Perform. Comput. Appl. 37, 2023). The basis vectors have unit norm and are stored
# scaled by s0 = √n (s0 = 1 in the working precision): their root mean square is then 1
# and their modulus at most √n, well within the range of Float16 (normal numbers from
# 6e-5 to 65504), so that only entries far below the rms lose relative precision. The
# stored basis is not exactly the computed one (relative rounding about 5e-4 in Float16):
# the estimated residual can then drift from the true one, and check_residual restarts
# from the true residual when they disagree, as an iterative refinement.
#
# Products with the basis: window_dots! and window_combine! (stored_window.jl, to be
# included first).
#
# Memory: memory + 1 stored vectors, 3 vectors in the working precision (x, w, v), and
# chunk × (memory + deflation + 2) entries for the restart.
# ---------------------------------------------------------------------------------------

"""
    GmresDRWorkspace(m, n, S; memory = 40, deflation = 20, chunk = 2^12,
                     check_residual = true, storage = nothing)

- memory: maximum dimension of the Krylov basis, i.e. restart length;
- deflation: number of harmonic Ritz vectors kept at each restart (0: restarted GMRES;
  memory larger than the number of products: full GMRES);
- chunk: number of rows of the basis processed at once at a restart (buffer size);
- check_residual: when the estimated residual reaches the tolerance, compute the true
  residual b - A x (one more product); if it is still too large, restart from it;
- storage: element type of the stored basis, e.g. Complex{Float16}; nothing keeps the
  working precision.

The residual estimates of the iterations of the last solve are in `residuals`, and the
number of restarts from the true residual (estimate converged, true residual not) in
`refinements[]`.
"""
struct GmresDRWorkspace{T, R, S, M, B}
    x::S                          # iterate, then solution
    w::S                          # new Arnoldi vector, true residual b - A x
    v::S                          # current basis vector in the working precision
    C::B                          # stored Krylov basis s0·V, n × (memory + 1)
    s0::R                         # storage scale
    buffers::WindowBuffers{S, M}  # device buffers of the products with C
    Pd::M                         # restart matrix P on the device
    chunk_in::M                   # rows of V, chunk × (memory + 1)
    chunk_out::M                  # rows of V·P, chunk × (deflation + 1)
    H::Matrix{T}                  # A·V[:, 1:j] = V[:, 1:j+1]·H[1:j+1, 1:j]
    Rq::Matrix{T}                 # H reduced to upper triangular form
    c::Vector{T}                  # Vᴴ·r at the start of the cycle
    g::Vector{T}                  # c rotated like Rq: |g[j+1]| is the residual norm
    y::Vector{T}                  # weights of the basis vectors, x += V·y
    s::Vector{T}                  # residual c - H·y of the projected problem
    hh::Vector{T}                 # host coefficients, memory + 1
    t::Vector{T}                  # host buffer, deflation + 1
    Q0::Matrix{T}                 # Q of the leading block after a restart
    P::Matrix{T}                  # restart matrix, (memory + 1) × (deflation + 1)
    rot::Vector{LinearAlgebra.Givens{T}}
    residuals::Vector{R}
    refinements::Base.RefValue{Int}
    memory::Int
    deflation::Int
    chunk::Int
    check_residual::Bool
    stats::IterativeStats
end

function GmresDRWorkspace(m, n, S::Type; memory = 40, deflation = 20, chunk = 2^12,
                          check_residual = true, storage = nothing)
    0 ≤ deflation < memory || throw(ArgumentError("0 ≤ deflation < memory required"))
    x = S(undef, n)
    T = eltype(x)
    R = real(T)
    T <: Complex || throw(ArgumentError("complex element type required"))
    Tc = isnothing(storage) ? T : storage
    Tc <: Complex || throw(ArgumentError("complex storage type required"))
    s0 = Tc == T ? one(R) : sqrt(R(n))
    C = similar(x, Tc, n, memory + 1)
    k1, rows = deflation + 1, min(chunk, n)
    GmresDRWorkspace(x, similar(x), similar(x), C, s0, WindowBuffers(x, Tc, memory + 1),
                     similar(x, memory + 1, k1),
                     similar(x, rows, memory + 1), similar(x, rows, k1),
                     zeros(T, memory + 1, memory), zeros(T, memory + 1, memory),
                     zeros(T, memory + 1), zeros(T, memory + 1), zeros(T, memory),
                     zeros(T, memory + 1), zeros(T, memory + 1), zeros(T, k1),
                     zeros(T, k1, k1), zeros(T, memory + 1, k1),
                     Vector{LinearAlgebra.Givens{T}}(undef, memory), R[], Ref(0),
                     memory, deflation, chunk, check_residual, IterativeStats())
end

Krylov.solution(ws::GmresDRWorkspace) = ws.x

function GmresDRSolver(system::BidirectionalSystem; memory = 40, deflation = 20,
                       chunk = 2^12, check_residual = true, storage = nothing)
    FixedPointSolver(system, GmresDRWorkspace; memory, deflation, chunk, check_residual,
                     storage)
end

# ---------------------------------------------------------------------------------------
# Stored basis
# ---------------------------------------------------------------------------------------

# Basis vector j in the working precision, into v
function load_column!(v, ws::GmresDRWorkspace, j)
    v .= view(ws.C, :, j) ./ ws.s0
end

# Unit vector v stored as basis vector j (rounded to the storage precision)
function store_column!(ws::GmresDRWorkspace, j, v)
    view(ws.C, :, j) .= v .* ws.s0
    ws
end

# hh[1:j] = C[:, 1:j]ᴴ·w on the host, i.e. s0 times the components of w on the basis
basis_dots!(ws::GmresDRWorkspace, w, j) = window_dots!(ws.hh, ws.buffers, ws.C, w, j)

# target += C[:, 1:j]·a, with the coefficients a = hh[1:j] given on the host
function basis_combine!(ws::GmresDRWorkspace, target, j)
    window_combine!(target, ws.buffers, ws.C, ws.hh, j)
end

# ---------------------------------------------------------------------------------------
# Iterations
# ---------------------------------------------------------------------------------------

# Arnoldi step: w ← A·V[:, j], orthogonalized against V[:, 1:j] by two passes of
# classical Gram–Schmidt, normalized and stored as V[:, j+1]; coefficients in
# H[1:j+1, j]. Returns false on a breakdown (A·V[:, j] in the span of the basis: the
# projected problem is then exact).
function arnoldi_step!(ws::GmresDRWorkspace, op, j)
    w, H, s0 = ws.w, ws.H, ws.s0
    mul!(w, op, load_column!(ws.v, ws, j))
    h, hh = view(H, 1:j, j), view(ws.hh, 1:j)
    fill!(h, zero(eltype(h)))
    for pass in 1:2
        basis_dots!(ws, w, j)                   # hh = s0·Vᴴ·w
        hh ./= s0
        h .+= hh
        hh .*= -inv(s0)                         # w -= V·(Vᴴ·w) = C·(-Vᴴ·w / s0)
        basis_combine!(ws, w, j)
    end
    nv = norm(w)
    H[j + 1, j] = nv
    iszero(nv) && return false
    w ./= nv
    store_column!(ws, j + 1, w)
    true
end

# Start of a cycle with k0 kept vectors: QR factorization of the dense leading block
# H[1:k0+1, 1:k0] (Q0), applied to c. Without kept vector, g = c = (‖r‖, 0, ...).
function start_cycle!(ws::GmresDRWorkspace, k0)
    T = eltype(ws.H)
    fill!(ws.Rq, zero(T))
    copyto!(ws.g, ws.c)
    k0 == 0 && return
    a = 1:(k0 + 1)
    lead, Q0, t = view(ws.H, a, 1:k0), view(ws.Q0, a, a), view(ws.t, a)
    fill!(Q0, zero(T))
    for i in a
        Q0[i, i] = one(T)
    end
    lmul!(qr(lead).Q, Q0)                       # small, on the host
    mul!(view(ws.Rq, a, 1:k0), Q0', lead)
    mul!(t, Q0', view(ws.c, a))
    copyto!(view(ws.g, a), t)
    nothing
end

# Column j of H brought to upper triangular form in Rq: Q0ᴴ on the rows of the leading
# block, the rotations of the previous Arnoldi columns, then a new rotation that zeroes
# the subdiagonal entry, also applied to g.
function reduce_column!(ws::GmresDRWorkspace, j, k0)
    col = view(ws.Rq, 1:(j + 1), j)
    copyto!(col, view(ws.H, 1:(j + 1), j))
    if k0 > 0
        a = 1:(k0 + 1)
        lead, t = view(col, a), view(ws.t, a)
        mul!(t, view(ws.Q0, a, a)', lead)
        copyto!(lead, t)
    end
    for i in (k0 + 1):(j - 1)
        lmul!(ws.rot[i], col)
    end
    G, r = givens(col[j], col[j + 1], j, j + 1)
    col[j], col[j + 1] = r, zero(r)
    ws.rot[j] = G
    lmul!(G, view(ws.g, 1:(j + 1)))
    abs(ws.g[j + 1])
end

# End of a cycle of j columns: weights y = Rq⁻¹·g, x += V·y, and residual of the projected
# problem s = c - H·y, i.e. b - A x = V[:, 1:j+1]·s.
function update_solution!(ws::GmresDRWorkspace, j)
    y = view(ws.y, 1:j)
    copyto!(y, view(ws.g, 1:j))
    ldiv!(UpperTriangular(view(ws.Rq, 1:j, 1:j)), y)
    view(ws.hh, 1:j) .= y ./ ws.s0
    basis_combine!(ws, ws.x, j)                 # x += V·y = C·(y / s0)
    s = view(ws.s, 1:(j + 1))
    copyto!(s, view(ws.c, 1:(j + 1)))
    mul!(s, view(ws.H, 1:(j + 1), 1:j), y, -1, true)
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

# Restart after a full cycle: P = [harmonic Ritz vectors (padded with a zero), residual s],
# orthonormalized; new basis V·P, new H = Pᴴ·H·P[1:memory, 1:k], new c = Pᴴ·s. Harmonic
# Ritz vectors too close to the span of the previous ones are skipped. Returns the number
# k of vectors kept (≤ deflation).
function deflate!(ws::GmresDRWorkspace)
    M, H, P = ws.memory, ws.H, ws.P
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
    s = view(ws.s, 1:(M + 1))
    p = view(P, :, k + 1)
    copyto!(p, s)
    orthogonalize!(p, view(P, :, 1:k))
    p ./= norm(p)
    Pk = view(P, :, 1:(k + 1))
    Hn = Pk' * (H * view(P, 1:M, 1:k))          # (k + 1) × k, small
    cn = Pk' * s
    fill!(H, zero(T))
    H[1:(k + 1), 1:k] .= Hn
    fill!(ws.c, zero(T))
    ws.c[1:(k + 1)] .= cn
    rotate_basis!(ws, k)
    k
end

# V[:, 1:k+1] ← V·P[:, 1:k+1] in place, by blocks of rows computed in the working
# precision, then the last kept vector is reorthogonalized against the k first ones
# (rounding errors only).
function rotate_basis!(ws::GmresDRWorkspace, k)
    C, s0, n = ws.C, ws.s0, size(ws.C, 1)
    copyto!(ws.Pd, ws.P)
    Pk = view(ws.Pd, :, 1:(k + 1))
    for first in 1:ws.chunk:n
        rows = first:min(first + ws.chunk - 1, n)
        rows_in = view(ws.chunk_in, 1:length(rows), :)
        rows_out = view(ws.chunk_out, 1:length(rows), 1:(k + 1))
        rows_in .= view(C, rows, :) ./ s0
        mul!(rows_out, rows_in, Pk)
        view(C, rows, 1:(k + 1)) .= rows_out .* s0
    end
    k == 0 && return
    v, hh = load_column!(ws.v, ws, k + 1), view(ws.hh, 1:k)
    basis_dots!(ws, v, k)                       # hh = s0·Vᴴ·v
    hh .*= -inv(s0)^2                           # v -= V·(Vᴴ·v) = C·(-s0·Vᴴ·v / s0²)
    basis_combine!(ws, v, k)
    v ./= norm(v)
    store_column!(ws, k + 1, v)
    nothing
end

"""Solve A x = b from x = 0 by GMRES-DR; at most maxit products by A."""
function gmresdr!(ws::GmresDRWorkspace, op, b; atol, rtol, maxit)
    x, w, M = ws.x, ws.w, ws.memory
    T = eltype(x)
    fill!(x, zero(T))
    copyto!(w, b)                               # residual of x = 0, without a product
    rnorm = norm(w)
    tol = max(atol, rtol * norm(b))
    empty!(ws.residuals)
    push!(ws.residuals, rnorm)
    ws.refinements[] = 0
    niter, k0 = 0, 0
    fresh = true                                # cycle started from the true residual
    converged = rnorm ≤ tol
    while !converged && niter < maxit
        if fresh
            k0 = 0
            fill!(ws.H, zero(T))
            fill!(ws.c, zero(T))
            ws.c[1] = rnorm
            w ./= rnorm
            store_column!(ws, 1, w)
        end
        start_cycle!(ws, k0)
        j, estimate, ok = k0, Inf, true
        while j < M && niter < maxit
            j += 1
            ok = arnoldi_step!(ws, op, j)
            niter += 1
            estimate = reduce_column!(ws, j, k0)
            push!(ws.residuals, estimate)
            (estimate ≤ tol || !ok) && break
        end
        update_solution!(ws, j)
        if estimate ≤ tol || !ok
            if ws.check_residual && niter < maxit
                mul!(w, op, x)
                niter += 1
                w .= b .- w
                rnorm = norm(w)
                push!(ws.residuals, rnorm)
                converged = rnorm ≤ tol
                converged || (ws.refinements[] += 1)
                fresh = true                    # restart from the true residual
            else
                converged = estimate ≤ tol
            end
        elseif j == M && niter < maxit
            k0 = deflate!(ws)
            fresh = false
        end
    end
    ws.stats.niter = niter
    ws.stats.solved = converged
    ws.stats
end

function krylov_solve!(solver::BidirectionalSolver{<:GmresDRWorkspace},
                       op::LinearOperator; atol = 0, rtol = default_rtol(solver),
                       itmax = 0, kwargs...)
    gmresdr!(solver.workspace, op, getdata(solver.r); atol, rtol,
             maxit = max_products(op, itmax))
end
