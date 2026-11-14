# ---------------------------------------------------------------------------------------
# GCROT
#
# GCRO with truncation (de Sturler, SIAM J. Numer. Anal. 36, 1999; Hicken & Zingg, SIAM
# J. Sci. Comput. 32, 2010). An outer GCR-type loop keeps a space of `outer` directions
# U with images C = A·U, C orthonormal, and the residual orthogonal to C: x is optimal
# over span(U). Each outer iteration runs `inner` steps of GMRES on the projected
# operator (I - C·Cᴴ)·A, which does not redo the work of the outer space:
# A·V = C·B + V⁺·H̄. The inner solution gives a new direction u = V·y - U·B·y, with
# image c = A·u = V⁺·H̄·y obtained without any product by the operator, orthogonal to C.
# It joins the outer space; when the space is full, one direction is dropped, which keeps
# C orthonormal.
#
# Truncation. 'oldest' drops the oldest direction. 'usage' drops the direction least used
# by the recent inner corrections: the weight of c_i in the correction is |(B·y)_i| (the
# part of A·V·y along c_i, removed by the term -U·B·y), accumulated as
# score_i ← decay·score_i + |(B·y)_i|; a new direction is kept at least one iteration.
# Directions that the inner solves keep correcting along, typically the slow modes, are
# then kept, whatever their age.
#
# Each outer iteration starts from the true residual b - A x (one product), projected out
# of span(C), so that rounding in a reduced storage of the vectors (`storage`) cannot
# make the reported residual drift. Per outer iteration: at most inner + 1 products.
# Orthogonalization: two passes of classical Gram–Schmidt against C, then against V.
# Memory: inner + 1 inner basis vectors and 2·outer outer vectors, stored, and 3 vectors
# in the working precision. Needs stored_window.jl.
# ---------------------------------------------------------------------------------------

"""
    GcrotWorkspace(m, n, S; inner = 30, outer = 30, truncation = :usage, decay = 0.5,
                   storage = nothing)

- inner: GMRES steps on the projected operator per outer iteration (products);
- outer: directions kept in the outer space;
- truncation: direction dropped when the outer space is full, :usage (least used by the
  recent inner corrections) or :oldest;
- decay: memory of the usage scores, from 0 (last correction only) to 1 (all of them);
- storage: element type of the stored vectors, e.g. Complex{Float16}; nothing keeps the
  working precision.

The residual estimates of the inner iterations and the true residuals at the start of
each outer iteration are in `residuals`.
"""
struct GcrotWorkspace{T, R, S, M, B}
    x::S                          # iterate, then solution
    w::S                          # residual, new Arnoldi vector, c
    v::S                          # basis vector in the working precision, u
    V::B                          # inner orthonormal basis, n × (inner + 1), as s0·v
    C::B                          # outer images, orthonormal, n × outer, as s0·c
    U::B                          # outer directions, A·U = C, n × outer, scales su
    s0::R                         # scale of the stored unit vectors
    su::Vector{R}
    buffers::WindowBuffers{S, M}
    H::Matrix{T}                  # inner Hessenberg matrix, (inner + 1) × inner
    Bm::Matrix{T}                 # Cᴴ·A·V, outer × inner
    Rq::Matrix{T}                 # H reduced to upper triangular form
    g::Vector{T}                  # ‖r‖·e₁ rotated like Rq
    y::Vector{T}                  # inner weights
    hy::Vector{T}                 # H̄·y: weights of V for c
    cy::Vector{T}                 # B·y, and components Cᴴ·r
    hh::Vector{T}                 # host coefficients
    rot::Vector{LinearAlgebra.Givens{T}}
    score::Vector{R}              # usage of the outer directions (truncation :usage)
    residuals::Vector{R}
    inner::Int
    outer::Int
    truncation::Symbol
    decay::Float64
    stats::IterativeStats
end

function GcrotWorkspace(m, n, S::Type; inner = 30, outer = 30, truncation = :usage,
                        decay = 0.5, storage = nothing)
    inner ≥ 1 && outer ≥ 0 || throw(ArgumentError("inner ≥ 1 and outer ≥ 0 required"))
    truncation in (:usage, :oldest) ||
        throw(ArgumentError("truncation must be :usage or :oldest"))
    x = S(undef, n)
    T = eltype(x)
    R = real(T)
    T <: Complex || throw(ArgumentError("complex element type required"))
    Tc = isnothing(storage) ? T : storage
    Tc <: Complex || throw(ArgumentError("complex storage type required"))
    nw = max(inner + 1, outer)              # widest window product
    GcrotWorkspace(x, similar(x), similar(x), similar(x, Tc, n, inner + 1),
                   similar(x, Tc, n, outer), similar(x, Tc, n, outer),
                   unit_scale(Tc, T, n), ones(R, outer), WindowBuffers(x, Tc, nw),
                   zeros(T, inner + 1, inner), zeros(T, outer, inner),
                   zeros(T, inner + 1, inner), zeros(T, inner + 1), zeros(T, inner),
                   zeros(T, inner + 1), zeros(T, outer), zeros(T, nw),
                   Vector{LinearAlgebra.Givens{T}}(undef, inner), fill(R(Inf), outer),
                   R[], inner, outer, truncation, Float64(decay), IterativeStats())
end

Krylov.solution(ws::GcrotWorkspace) = ws.x

function GcrotSolver(system::BidirectionalSystem; inner = 30, outer = 30,
                     truncation = :usage, decay = 0.5, storage = nothing)
    FixedPointSolver(system, GcrotWorkspace; inner, outer, truncation, decay, storage)
end

"""Solve A x = b from x = 0 by GCROT; at most maxit products by A."""
function gcrot!(ws::GcrotWorkspace, op, b; atol, rtol, maxit)
    x, w, v, V, C, U = ws.x, ws.w, ws.v, ws.V, ws.C, ws.U
    H, Bm, g, hh, buffers = ws.H, ws.Bm, ws.g, ws.hh, ws.buffers
    T = eltype(x)
    m, k, s0 = ws.inner, ws.outer, ws.s0
    fill!(x, zero(T))
    copyto!(w, b)                               # residual of x = 0, without a product
    rnorm = norm(w)
    tol = max(atol, rtol * norm(b))
    empty!(ws.residuals)
    push!(ws.residuals, rnorm)
    niter, nc, slot = 0, 0, 1                   # outer directions in use, next slot
    score = ws.score
    fill!(score, Inf)
    estimate = rnorm
    converged = rnorm ≤ tol
    while !converged && niter < maxit
        # Residual projected out of span(C): γ = Cᴴ·r, x += U·γ, r -= C·γ. If that is
        # enough, the true residual confirms it (in reduced storage, A·U = C holds only
        # to rounding); otherwise the inner solve starts from it, unprojected.
        if nc > 0
            γ = view(ws.cy, 1:nc)
            window_dots!(hh, buffers, C, w, nc)
            γ .= view(hh, 1:nc) ./ s0
            view(hh, 1:nc) .= γ ./ view(ws.su, 1:nc)
            window_combine!(x, buffers, U, hh, nc)
            view(hh, 1:nc) .= .-γ ./ s0
            window_combine!(w, buffers, C, hh, nc)
            rnorm = norm(w)
            if rnorm ≤ tol
                mul!(w, op, x)
                niter += 1
                w .= b .- w
                rnorm = norm(w)
                push!(ws.residuals, rnorm)
                converged = rnorm ≤ tol
                (converged || niter ≥ maxit) && break
            end
        end
        # Inner GMRES on (I - C·Cᴴ)·A from the projected residual
        fill!(H, zero(T))
        fill!(Bm, zero(T))
        fill!(g, zero(T))
        g[1] = rnorm
        w ./= rnorm
        store_scaled!(V, 1, w, s0)
        j = 0
        while j < m && niter < maxit
            j += 1
            mul!(w, op, load_stored!(v, V, j, s0))
            niter += 1
            orthogonalize_window!(w, view(Bm, 1:nc, j), hh, buffers, C, s0, nc)
            orthogonalize_window!(w, view(H, 1:j, j), hh, buffers, V, s0, j)
            nv = norm(w)
            H[j + 1, j] = nv
            estimate = hessenberg_column!(ws.Rq, g, ws.rot, H, j)
            push!(ws.residuals, estimate)
            # V[:, j+1] is needed by c below, even if the inner solve stops here
            if iszero(nv)
                fill!(view(V, :, j + 1), zero(eltype(V)))
            else
                w ./= nv
                store_scaled!(V, j + 1, w, s0)
            end
            (estimate ≤ tol || iszero(nv)) && break
        end
        j == 0 && break
        y = hessenberg_solve!(ws.y, ws.Rq, g, j)
        # u = V[:, 1:j]·y - U·(B·y), into v; c = A·u = V[:, 1:j+1]·H̄·y, into w
        fill!(v, zero(T))
        view(hh, 1:j) .= y ./ s0
        window_combine!(v, buffers, V, hh, j)
        if nc > 0
            by = view(ws.cy, 1:nc)
            mul!(by, view(Bm, 1:nc, 1:j), y)
            for i in 1:nc                       # usage of the outer directions
                score[i] = isinf(score[i]) ? abs(by[i]) :
                           ws.decay * score[i] + abs(by[i])
            end
            view(hh, 1:nc) .= .-by ./ view(ws.su, 1:nc)
            window_combine!(v, buffers, U, hh, nc)
        end
        hy = view(ws.hy, 1:(j + 1))
        mul!(hy, view(H, 1:(j + 1), 1:j), y)
        view(hh, 1:(j + 1)) .= hy ./ s0
        fill!(w, zero(T))
        window_combine!(w, buffers, V, hh, j + 1)
        x .+= v
        if k > 0
            # c reorthogonalized against C (rounding only), u alike to keep A·u = c
            if nc > 0
                δ = view(ws.cy, 1:nc)
                window_dots!(hh, buffers, C, w, nc)
                δ .= view(hh, 1:nc) ./ s0
                view(hh, 1:nc) .= .-δ ./ s0
                window_combine!(w, buffers, C, hh, nc)
                view(hh, 1:nc) .= .-δ ./ view(ws.su, 1:nc)
                window_combine!(v, buffers, U, hh, nc)
            end
            # New outer direction with unit ‖c‖, in the next slot (oldest dropped)
            α = norm(w)
            if !iszero(α)
                w ./= α
                v ./= α
                store_scaled!(C, slot, w, s0)
                ws.su[slot] = storage_scale(U, v)
                store_scaled!(U, slot, v, ws.su[slot])
                score[slot] = Inf               # kept at least until its first use
                nc = min(nc + 1, k)
                # Next slot: the next empty one, then the one to drop
                slot = nc < k || ws.truncation == :oldest ? mod1(slot + 1, k) :
                       argmin(score)
            end
        end
        # True residual, start of the next outer iteration
        if niter < maxit
            mul!(w, op, x)
            niter += 1
            w .= b .- w
            rnorm = norm(w)
            push!(ws.residuals, rnorm)
            converged = rnorm ≤ tol
        else
            converged = estimate ≤ tol
        end
    end
    ws.stats.niter = niter
    ws.stats.solved = converged
    ws.stats
end

function krylov_solve!(solver::BidirectionalSolver{<:GcrotWorkspace},
                       op::LinearOperator; atol = 0, rtol = default_rtol(solver),
                       itmax = 0, kwargs...)
    gcrot!(solver.workspace, op, getdata(solver.r); atol, rtol,
           maxit = max_products(op, itmax))
end
