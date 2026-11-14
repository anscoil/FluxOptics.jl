# ---------------------------------------------------------------------------------------
# Anderson acceleration
#
# Type II Anderson acceleration (Walker & Ni, SIAM J. Numer. Anal. 49, 2011) of the
# Richardson map g(x) = x + (b - A x), whose residual is f(x) = b - A x. With A = I - K,
# g is the plain fixed-point map x ↦ K x + b. With a window larger than the number of
# iterations, the iterates are those of GMRES followed by one more step of g.
#
# The last `memory` differences of iterates (dX) and of residuals (dF) are kept in the
# slots of a ring buffer. The least-squares problem min ‖f - dF·γ‖ is solved through the
# Gram matrix dFᴴ·dF, updated by one row and one column per iteration and regularized,
# rather than by a QR factorization: dropping the oldest difference then costs nothing,
# and every vector operation is a product with the n × memory window.
# Per iteration: one product by the operator and four products with the window.
#
# Storage. The differences can be stored in a lower precision than the working one
# (`storage`, e.g. Complex{Float16}), through window_dots! and window_combine!
# (stored_window.jl, to be included first). Each stored difference is scaled by its own
# factor √n/‖·‖ (root mean square 1), kept on the host: differences shrink with the
# residual, and would otherwise fall below the range of Float16. The Gram matrix is that
# of the stored differences, so that the least-squares problem stays consistent. The
# residual f = b - A x is always computed in the working precision from the new iterate,
# so that the reduced storage only makes the extrapolation less accurate: it cannot make
# the reported residual wrong.
#
# Memory: 2·memory stored vectors and 5 vectors in the working precision.
# ---------------------------------------------------------------------------------------

"""
    AndersonWorkspace(m, n, S; memory = 20, beta = 1.0, reg = 1e-10,
                      restart_growth = Inf, keep_best = true, storage = nothing)

- memory: number of differences kept (window of the least-squares problem);
- beta: mixing, the new iterate is x̄ + beta·f̄ (beta = 1: x̄ + f̄ = g(x̄));
- reg: Tikhonov regularization of the Gram matrix, relative to its mean diagonal; if the
  regularized matrix still fails to factorize, the history is cleared;
- restart_growth: the history is cleared when the residual grows by this factor from one
  iteration to the next (Inf: never);
- keep_best: return the iterate of smallest residual if the solve does not converge;
- storage: element type of the stored differences, e.g. Complex{Float16}; nothing keeps
  the working precision.
"""
struct AndersonWorkspace{T, R, S, M, B}
    x::S                          # current iterate, then solution
    f::S                          # residual b - A x
    w::S                          # A x, then the new residual difference
    d::S                          # step x_{k+1} - x_k
    best::S                       # iterate of smallest residual, saved on regressions
    dX::B                         # stored differences of iterates, ring buffer
    dF::B                         # stored differences of residuals, same slots
    sx::Vector{R}                 # storage scale of each slot of dX
    sf::Vector{R}                 # storage scale of each slot of dF
    buffers::WindowBuffers{S, M}  # device buffers of the products with the window
    G::Matrix{T}                  # Gram matrix dFᴴ·dF of the slots in use
    Gr::Matrix{T}                 # regularized copy, factorized in place
    c::Vector{T}                  # dFᴴ·f, then the weights γ (host)
    t::Vector{T}                  # coefficients of the window products (host)
    memory::Int
    beta::Float64
    reg::Float64
    restart_growth::Float64
    keep_best::Bool
    stats::IterativeStats
end

function AndersonWorkspace(m, n, S::Type; memory = 20, beta = 1.0, reg = 1e-10,
                           restart_growth = Inf, keep_best = true, storage = nothing)
    x = S(undef, n)
    T = eltype(x)
    R = real(T)
    Tc = isnothing(storage) ? T : storage
    Tc <: Complex || throw(ArgumentError("complex storage type required"))
    AndersonWorkspace(x, similar(x), similar(x), similar(x), similar(x),
                      similar(x, Tc, n, memory), similar(x, Tc, n, memory),
                      ones(R, memory), ones(R, memory), WindowBuffers(x, Tc, memory),
                      zeros(T, memory, memory), zeros(T, memory, memory), zeros(T, memory),
                      zeros(T, memory), memory, Float64(beta), Float64(reg),
                      Float64(restart_growth), keep_best, IterativeStats())
end

Krylov.solution(ws::AndersonWorkspace) = ws.x

function AndersonSolver(system::BidirectionalSystem; memory = 20, beta = 1.0, reg = 1e-10,
                        restart_growth = Inf, keep_best = true, storage = nothing)
    FixedPointSolver(system, AndersonWorkspace; memory, beta, reg, restart_growth,
                     keep_best, storage)
end

# Stores v in slot of C with its scale (1 in the working precision, √n/‖v‖ otherwise)
function store_difference!(C, scales, slot, v)
    s = one(eltype(scales))
    if eltype(C) != eltype(v)
        nv = norm(v)
        iszero(nv) || (s = sqrt(eltype(scales)(length(v))) / nv)
    end
    scales[slot] = s
    view(C, :, slot) .= v .* s
end

# Stored difference of a slot in the working precision: the column itself in the
# working precision, otherwise its rescaled copy in buffer
function stored_difference(buffer, C, scales, slot)
    eltype(C) == eltype(buffer) && return view(C, :, slot)
    buffer .= view(C, :, slot) ./ scales[slot]
end

# Step d = beta·f - (dX + beta·dF)·γ, with γ = (dFᴴ·dF + regularization)⁻¹·dFᴴ·f over the
# nact slots in use. Returns false if the Gram matrix cannot be factorized; d is then the
# plain relaxed step beta·f.
function anderson_step!(ws::AndersonWorkspace, nact)
    ws.d .= ws.beta .* ws.f
    nact == 0 && return true
    a = 1:nact
    c, t = view(ws.c, a), view(ws.t, a)
    window_dots!(ws.c, ws.buffers, ws.dF, ws.f, nact)    # stored columns: scales sf
    c ./= view(ws.sf, a)
    Gr = view(ws.Gr, a, a)
    copyto!(Gr, view(ws.G, a, a))
    shift = ws.reg * real(tr(Gr)) / nact
    for i in a
        Gr[i, i] += shift
    end
    factor = cholesky!(Hermitian(Gr); check = false)
    issuccess(factor) || return false
    ldiv!(factor, c)
    t .= .-c ./ view(ws.sx, a)                          # d -= dX·γ
    window_combine!(ws.d, ws.buffers, ws.dX, ws.t, nact)
    t .= -ws.beta .* c ./ view(ws.sf, a)                # d -= beta·dF·γ
    window_combine!(ws.d, ws.buffers, ws.dF, ws.t, nact)
    true
end

"""Solve A x = b from x = 0 by Anderson acceleration; at most maxit products by A."""
function anderson!(ws::AndersonWorkspace, op, b; atol, rtol, maxit)
    x, f, w, d = ws.x, ws.f, ws.w, ws.d
    fill!(x, zero(eltype(x)))
    copyto!(f, b)                     # residual of x = 0, without a product
    tol = max(atol, rtol * norm(b))
    nact, slot, niter = 0, 1, 0
    nf, nf_prev = norm(f), Inf
    best_norm = nf
    ws.keep_best && fill!(ws.best, zero(eltype(x)))
    while nf > tol && niter < maxit
        if !anderson_step!(ws, nact)
            nact, slot = 0, 1                         # clear the history
        end
        x .+= d
        mul!(w, op, x)                                # the only product by the operator
        niter += 1
        # New differences in the current slot: dx = d, df = f_new - f_old
        w .= b .- w .- f
        f .+= w
        store_difference!(ws.dX, ws.sx, slot, d)
        store_difference!(ws.dF, ws.sf, slot, w)
        nact = min(nact + 1, ws.memory)
        # Row and column of the new slot in the Gram matrix of the stored differences
        a = 1:nact
        c = view(ws.c, a)
        df = stored_difference(w, ws.dF, ws.sf, slot)
        window_dots!(ws.c, ws.buffers, ws.dF, df, nact)
        c ./= view(ws.sf, a)
        for j in a
            ws.G[j, slot] = c[j]
            ws.G[slot, j] = conj(c[j])
        end
        slot = mod1(slot + 1, ws.memory)
        nf_prev, nf = nf, norm(f)
        # On a regression, the previous iterate x - d may be the best one so far
        if ws.keep_best && nf > nf_prev && nf_prev < best_norm
            best_norm = nf_prev
            ws.best .= x .- d
        end
        if nf > ws.restart_growth * nf_prev
            nact, slot = 0, 1
        end
    end
    converged = nf ≤ tol
    !converged && ws.keep_best && best_norm < nf && copyto!(x, ws.best)
    ws.stats.niter = niter
    ws.stats.solved = converged
    ws.stats
end

function krylov_solve!(solver::BidirectionalSolver{<:AndersonWorkspace},
                       op::LinearOperator; atol = 0, rtol = default_rtol(solver),
                       itmax = 0, kwargs...)
    anderson!(solver.workspace, op, getdata(solver.r); atol, rtol,
              maxit = max_products(op, itmax))
end
