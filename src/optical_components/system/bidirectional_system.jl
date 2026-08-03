struct BidirectionalSystem{S1, S2, C, P, K, J}
    s_in::S1
    s_out::S2
    s_in_adj::S1
    s_out_adj::S2
    s_in_zero::S1
    s_out_zero::S2
    components::C
    fp_state::P
    fp_state_adj::P
    tmp_state::P
    fp_state_keys::K
    spectral_projector::J
end

Functors.@functor BidirectionalSystem (components,)

_interleave(ifaces::Tuple{T}, ::Tuple{}) where {T} = ifaces
_interleave(ifaces, comps) =
    (first(ifaces), first(comps), _interleave(Base.tail(ifaces), Base.tail(comps))...)

coalesce_state(s) = fmap(x -> isnothing(x) ? [] : collect(x), s)

function state_view(fp_state::ComponentArray, k::Symbol)
    v = getproperty(fp_state, k)
    if Functors.isleaf(v) && isempty(v)
        nothing
    else
        fmap(x -> isempty(x) ? nothing : x, NamedTuple(v))
    end
end

real_n(n0::Number) = real(n0)
real_n(::Nothing) = 0

function BidirectionalSystem(s_in::AbstractBidirectionalSource{U},
                             s_out::AbstractBidirectionalSource{U},
                             components::Vararg{AbstractBidirectionalComponent}) where {U}
    s_in_adj = zero(s_in)
    s_out_adj = zero(s_out)
    s_in_zero = zero(s_in)
    s_out_zero = zero(s_out)
    u0 = get_source(s_in)
    all_nodes = (s_in, components..., s_out)
    flat_interfaces = map(
        (l, r) -> FlatInterface(u0, get_n0_right(l), get_n0_left(r)),
        Base.front(all_nodes),
        Base.tail(all_nodes)
    )
    all_components = _interleave(flat_interfaces, components)
    state_keys = ntuple(i -> Symbol(:c, i), length(all_components))
    states = map(c -> coalesce_state(alloc_fp_state(u0, c)), all_components)
    fp_state = adapt(similar(U, 1), ComponentArray(NamedTuple{state_keys}(states)))
    fp_state .= 0
    fp_state_adj = copy(fp_state)
    tmp_state = copy(fp_state)
    n_max = max(real(get_n0(s_in)), real(get_n0(s_out)),
                map(c -> max(real_n(get_n0_left(c)), real_n(get_n0_right(c))), components)...)
    spectral_projector = real.(compute_kz(u0, n_max))
    @. spectral_projector = ifelse(spectral_projector > 0, 1, 0)
    BidirectionalSystem(s_in, s_out, s_in_adj, s_out_adj, s_in_zero, s_out_zero,
                        all_components, fp_state, fp_state_adj, tmp_state, state_keys,
                        spectral_projector)
end

function BidirectionalSystem(s_in::AbstractBidirectionalSource,
                             components::Vararg{AbstractBidirectionalComponent})
    s_out = zero(s_in; n0 = get_n0_right(Base.last(components)))
    BidirectionalSystem(s_in, s_out, components...)
end

function apply_spectral_projection!(s::BidirectionalSystem, fp_state::ComponentArray)
    ns = size(s.spectral_projector)
    state = reshape(getdata(fp_state), (ns..., length(fp_state) ÷ prod(ns)))
    @. state *= s.spectral_projector
end

function compute_roundtrip!(s::BidirectionalSystem,
                            s_in::AbstractBidirectionalSource,
                            s_out::AbstractBidirectionalSource,
                            fp_state::ComponentArray;
                            spectral_projection::Bool = false)
    if spectral_projection
        apply_spectral_projection!(s, fp_state)
    end
    fp_state_views = @ignore_derivatives map(k -> state_view(fp_state, k), s.fp_state_keys)
    u = propagate(s_in)
    for (p, state) in zip(s.components, fp_state_views)
        u = propagate!(u, state, p)
    end
    uf = propagate!(u, s_out)
    u = propagate(s_out)
    for (p, state) in zip(s.components[end:-1:1], reverse(fp_state_views))
        u = inverse_propagate!(u, state, p)
    end
    ur = inverse_propagate!(u, s_in)
    (uf, ur)
end

function compute_roundtrip_adjoint!(s::BidirectionalSystem,
                                    s_in_adj::AbstractBidirectionalSource,
                                    s_out_adj::AbstractBidirectionalSource,
                                    fp_state::ComponentArray;
                                    spectral_projection::Bool = false)
    fp_state_views = map(k -> state_view(fp_state, k), s.fp_state_keys)
    u = propagate(s_in_adj)
    u = inverse_propagate_adjoint!(u, s_in_adj)
    for (p, state) in zip(s.components, fp_state_views)
        u = inverse_propagate_adjoint!(u, state, p)
    end
    ∂uf = u
    u = propagate(s_out_adj)
    u = propagate_adjoint!(u, s_out_adj)
    for (p, state) in zip(s.components[end:-1:1], reverse(fp_state_views))
        u = propagate_adjoint!(u, state, p)
    end
    ∂ur = u
    if spectral_projection
        apply_spectral_projection!(s, fp_state)
    end
    (∂uf, ∂ur)
end

struct BidirectionalSolver{W, P, T}
    workspace::W
    r::P
    atol::Ref{T}
end

function compute_linear_operator(s::BidirectionalSystem;
                                 adjoint::Bool = false,
                                 spectral_projection::Bool = false)
    state = getdata(s.tmp_state)
    T = eltype(state)
    n = length(state)
    S = typeof(state)
    s_in, s_out = s.s_in_zero, s.s_out_zero
    function prod!(res, v, α, β)
        @. state = v
        compute_roundtrip!(s, s_in, s_out, s.tmp_state; spectral_projection)
        if iszero(β)
            @. res = α * (v - state)
        else
            @. res = α * (v - state) + β * res
        end
        res
    end
    function ctprod!(res, v, α, β)
        @. state = v
        compute_roundtrip_adjoint!(s, s_in, s_out, s.tmp_state; spectral_projection)
        if iszero(β)
            @. res = α * (v - state)
        else
            @. res = α * (v - state) + β * res
        end
        res
    end
    if adjoint
        LinearOperator(T, n, n, false, false, ctprod!, nothing, prod!; S)
    else
        LinearOperator(T, n, n, false, false, prod!, nothing, ctprod!; S)
    end
end

function FixedPointSolver(s::BidirectionalSystem, Workspace; kwargs...)
    r = similar(s.fp_state)
    v = getdata(r)
    n = length(v)
    S = typeof(v)
    ws = Workspace(n, n, S; kwargs...)
    T = real(eltype(v))
    BidirectionalSolver(ws, r, Ref(T(0.0)))
end

function GmresSolver(s::BidirectionalSystem; memory = 20)
    FixedPointSolver(s, GmresWorkspace; memory)
end

function BicgstabSolver(s::BidirectionalSystem)
    FixedPointSolver(s, BicgstabWorkspace)
end

function BilqSolver(s::BidirectionalSystem)
    FixedPointSolver(s, BilqWorkspace)
end

function CgneSolver(s::BidirectionalSystem)
    FixedPointSolver(s, CgneWorkspace)
end

function CraigSolver(s::BidirectionalSystem)
    FixedPointSolver(s, CraigWorkspace)
end

function krylov_solve!(solver::BidirectionalSolver{<:GmresWorkspace},
                       op::LinearOperator; kwargs...)
    gmres!(solver.workspace, op, getdata(solver.r); kwargs...)
end

function krylov_solve!(solver::BidirectionalSolver{<:BicgstabWorkspace},
                       op::LinearOperator; kwargs...)
    bicgstab!(solver.workspace, op, getdata(solver.r); kwargs...)
end

function krylov_solve!(solver::BidirectionalSolver{<:BilqWorkspace},
                       op::LinearOperator; kwargs...)
    bilq!(solver.workspace, op, getdata(solver.r); kwargs...)
end

function krylov_solve!(solver::BidirectionalSolver{<:CgneWorkspace},
                       op::LinearOperator; kwargs...)
    cgne!(solver.workspace, op, getdata(solver.r); kwargs...)
end

function krylov_solve!(solver::BidirectionalSolver{<:CraigWorkspace},
                       op::LinearOperator; kwargs...)
    craig!(solver.workspace, op, getdata(solver.r); kwargs...)
end

function fp_solve!(s::BidirectionalSystem, solver::BidirectionalSolver;
                   spectral_projection = false, n_warm_start = 0, kwargs...)
    if n_warm_start > 0
        fp_solve!(s; itmax = n_warm_start, spectral_projection)
    end
    s_in, s_out = s.s_in, s.s_out
    v0 = getdata(s.fp_state)
    vr = getdata(solver.r)
    @. vr = v0
    compute_roundtrip!(s, s_in, s_out, solver.r; spectral_projection)
    @. vr -= v0
    op = compute_linear_operator(s; spectral_projection)
    krylov_solve!(solver, op; atol = solver.atol[], kwargs...)
    res_state = Krylov.solution(solver.workspace)
    res = getdata(res_state)
    res = res isa Tuple ? first(res) : res
    @. v0 += res
    if solver.workspace.stats.solved && iszero(solver.atol[])
        @. vr = res
        compute_roundtrip!(s, s_in, s_out, solver.r; spectral_projection)
        @. vr -= res
        solver.atol[] = norm(vr)
    end
    s.fp_state
end

function fp_solve!(s::BidirectionalSystem;
                   itmax = 20, spectral_projection = false, kwargs...)
    s_in, s_out = s.s_in, s.s_out
    for i in 1:itmax
        compute_roundtrip!(s, s_in, s_out, s.fp_state; spectral_projection)
    end
    s.fp_state
end

function fp_solve!(s::BidirectionalSystem, ::Nothing; kwargs...)
    fp_solve!(s; itmax = 1, kwargs...)
end

function fp_solve_adjoint!(s::BidirectionalSystem, solver::BidirectionalSolver;
                           spectral_projection = false, n_warm_start = 0, kwargs...)
    if n_warm_start > 0
        fp_solve_adjoint!(s; itmax = n_warm_start, spectral_projection)
    end
    s_in, s_out = s.s_in_adj, s.s_out_adj
    v0 = getdata(s.fp_state_adj)
    vr = getdata(solver.r)
    @. vr = v0
    compute_roundtrip_adjoint!(s, s_in, s_out, solver.r; spectral_projection)
    @. vr -= v0
    op = compute_linear_operator(s; spectral_projection, adjoint = true)
    krylov_solve!(solver, op; kwargs...)
    res_state = Krylov.solution(solver.workspace)
    res = getdata(res_state)
    res = res isa Tuple ? first(res) : res
    @. v0 += res
    if solver.workspace.stats.solved && iszero(solver.atol[])
        @. vr = res
        compute_roundtrip_adjoint!(s, s_in, s_out, solver.r; spectral_projection)
        @. vr -= res
        solver.atol[] = norm(vr)
    end
    s.fp_state_adj
end

function fp_solve_adjoint!(s::BidirectionalSystem;
                           itmax = 20, spectral_projection = false, kwargs...)
    s_in, s_out = s.s_in_adj, s.s_out_adj
    for i in 1:itmax
        compute_roundtrip_adjoint!(s, s_in, s_out, s.fp_state_adj; spectral_projection)
    end
    s.fp_state_adj
end

function fp_solve_adjoint!(s::BidirectionalSystem, ::Nothing; kwargs...)
    fp_solve_adjoint!(s; itmax = 1, kwargs...)
end

function apply_implicit(uf, ur, s, solver; spectral_projection = false, kwargs...)
    (uf, ur)
end

function combine_implicit(uf, ur, ufi, uri)
    (uf, ur)
end

reset_state!(s::BidirectionalSystem, state::ComponentArray) = nothing

function propagate(s::BidirectionalSystem,
                   solver::Union{Nothing, BidirectionalSolver};
                   spectral_projection = false, kwargs...)
    fp_state = @ignore_derivatives fp_solve!(s, solver; spectral_projection, kwargs...)
    s_in, s_out = @ignore_derivatives s.s_in, s.s_out
    @ignore_derivatives copyto!(s.tmp_state, fp_state)
    uf, ur = compute_roundtrip!(s, s_in, s_out, s.tmp_state; spectral_projection)
    reset_state!(s, fp_state)
    ufi, uri = apply_implicit(uf, ur, s, solver; spectral_projection, kwargs...)
    uf, ur = combine_implicit(uf, ur, ufi, uri)
    # @ignore_derivatives fill!(s_in, ur)
    # @ignore_derivatives fill!(s_out, uf)
    (reflected = ur, transmitted = uf)
end

function test_adjoint(s, solver, rand!)
    s_in, s_out = s.s_in, s.s_out

    copyto!(solver.r, s.fp_state)
    compute_roundtrip!(s, s_in, s_out, solver.r)

    op = compute_linear_operator(s)

    v0 = getdata(s.fp_state)
    n = length(v0)
    S = typeof(v0)
    
    x = similar(v0); rand!(x)
    y = similar(v0); rand!(y)
    
    Cx = similar(x)
    op.prod!(Cx, x, one(eltype(x)), zero(eltype(x)))
    
    Cy = similar(y)
    op.ctprod!(Cy, y, one(eltype(y)), zero(eltype(y)))

    lhs = dot(Cx, y)
    rhs = dot(x, Cy)
    
    @show lhs
    @show rhs
    err = abs(lhs - rhs) / abs(lhs)
    @show err
    err
end
