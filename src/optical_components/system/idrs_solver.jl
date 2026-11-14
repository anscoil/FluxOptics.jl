"""Workspace of IDR(s), with s the dimension of the shadow space."""
struct IdrsWorkspace{S}
    x::S
    s::Int
    stats::IterativeStats
end

IdrsWorkspace(m, n, S::Type; s = 4) = IdrsWorkspace(S(undef, n), s, IterativeStats())

Krylov.solution(ws::IdrsWorkspace) = ws.x

IdrsSolver(system::BidirectionalSystem; s = 4) = FixedPointSolver(system, IdrsWorkspace; s)

# Keywords of Krylov.jl (atol, rtol, itmax) translated to those of IterativeSolvers;
# itmax = 0 sets no bound beyond the size of the problem
max_products(op, itmax) = itmax > 0 ? itmax : size(op, 2)
default_rtol(::BidirectionalSolver{W, P, T}) where {W, P, T} = sqrt(eps(T))

function krylov_solve!(solver::BidirectionalSolver{<:IdrsWorkspace},
                       op::LinearOperator; atol = 0, rtol = default_rtol(solver),
                       itmax = 0, kwargs...)
    ws = solver.workspace
    fill!(ws.x, zero(eltype(ws.x)))
    # idrs! has no initial_zero: one more product for the initial residual, not counted
    _, history = idrs!(ws.x, op, getdata(solver.r); s = ws.s, abstol = atol,
                       reltol = rtol, maxiter = max_products(op, itmax), log = true,
                       kwargs...)
    record!(ws.stats, history)
end
