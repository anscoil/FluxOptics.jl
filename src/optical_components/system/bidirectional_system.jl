struct BidirectionalSystem{S1, S2, C, P, K, J}
    start_source::S1
    end_source::S2
    components::C
    fp_state::P
    fp_state_keys::K
    spectral_projector::J
end

Functors.@functor BidirectionalSystem (components,)

_interleave(ifaces::Tuple{T}, ::Tuple{}) where {T} = ifaces
_interleave(ifaces, comps) =
    (first(ifaces), first(comps), _interleave(Base.tail(ifaces), Base.tail(comps))...)

coalesce_state(s) = s
coalesce_state(::Nothing) = []

function BidirectionalSystem(start_source::AbstractBidirectionalSource{U},
                             end_source::AbstractBidirectionalSource{U},
                             components::Vararg{AbstractBidirectionalComponent}) where {U}
    u0 = get_source(start_source)
    all_nodes = (start_source, components..., end_source)
    flat_interfaces = map(
        (l, r) -> FlatInterface(u0, get_n0_right(l), get_n0_left(r)),
        Base.front(all_nodes),
        Base.tail(all_nodes)
    )
    all_components = _interleave(flat_interfaces, components)
    state_keys = ntuple(i -> Symbol(:c, i), length(all_components))
    states = map(c -> coalesce_state(initial_state(u0, c)), all_components)
    fp_state = adapt(similar(U, 1), ComponentArray(NamedTuple{state_keys}(states)))
    n_max = max(real(get_n0(start_source)), real(get_n0(end_source)),
                map(c -> max(real(get_n0_left(c)), real(get_n0_right(c))), components)...)
    spectral_projector = real.(compute_kz(u0, n_max))
    @. spectral_projector = ifelse(spectral_projector > 0, 1, 0)
    BidirectionalSystem(start_source, end_source, all_components,
                        fp_state, state_keys, spectral_projector)
end

function BidirectionalSystem(start_source::AbstractBidirectionalSource,
                             components::Vararg{AbstractBidirectionalComponent})
    end_source = zero(start_source; n0 = get_n0_right(Base.last(components)))
    BidirectionalSystem(start_source, end_source, components...)
end

get_fp_state(s::BidirectionalSystem) = copy(s.fp_state)

set_fp_state!(s::BidirectionalSystem, fp_state) = copyto!(s.fp_state, fp_state)

function apply_spectral_projection!(s::BidirectionalSystem, fp_state::ComponentArray)
    nx, ny = size(s.spectral_projector)
    state = reshape(getdata(fp_state), (nx, ny, length(fp_state) ÷ (nx * ny)))
    @. state *= s.spectral_projector
end

function compute_roundtrip!(s::BidirectionalSystem, fp_state::ComponentArray;
                            spectral_projection::Bool = false)
    if spectral_projection
        apply_spectral_projection!(s, fp_state)
    end
    fp_state_views = map(k -> getproperty(fp_state, k), s.fp_state_keys)
    u = propagate(s.start_source)
    for (p, state) in zip(s.components, fp_state_views)
        u = propagate!(u, state, p)
    end
    u = propagate!(u, s.end_source)
    for (p, state) in zip(reverse(s.components), reverse(fp_state_views))
        u = inverse_propagate!(u, state, p)
    end
    u = inverse_propagate!(u, s.start_source)
    nothing
end

function compute_roundtrip_adjoint!(s::BidirectionalSystem, fp_state::ComponentArray;
                                    spectral_projection::Bool = false)
    if spectral_projection
        apply_spectral_projection!(s, fp_state)
    end
    fp_state_views = map(k -> getproperty(fp_state, k), s.fp_state_keys)
    u = propagate_zero(s.start_source)
    for (p, state) in zip(s.components, fp_state_views)
        u = inverse_propagate_adjoint!(u, state, p)
    end
    for (p, state) in zip(reverse(s.components), reverse(fp_state_views))
        u = propagate_adjoint!(u, state, p)
    end
    nothing
end

struct BidirectionalSolver{W, S, P}
    system::S
    workspace::W
    tmp_state::P
    r::P
end

function compute_linear_operator(s::BidirectionalSystem, tmp_state::ComponentArray, v0, r;
                                 spectral_projection::Bool = false)
    state = getdata(tmp_state)
    T = eltype(state)
    n = length(state)
    S = typeof(state)
    function prod!(res, v, α, β)
        @. state = v + v0
        compute_roundtrip!(s, tmp_state; spectral_projection)
        if iszero(β)
            @. res = α * (v + v0 + r - state)
        else
            @. res = α * (v + v0 + r - state) + β * res
        end
        res
    end
    function ctprod!(res, v, α, β)
        @. state = v
        compute_roundtrip_adjoint!(s, tmp_state; spectral_projection)
        if iszero(β)
            @. res = α * (v - state)
        else
            @. res = α * (v - state) + β * res
        end
        res
    end
    LinearOperator(T, n, n, false, false, prod!, nothing, ctprod!; S)
end

function compute_preconditioner(s::BidirectionalSystem, tmp_state::ComponentArray, v0, r)
    state = getdata(tmp_state)
    T = eltype(state)
    n = length(state)
    S = typeof(state)
    LinearOperator(T, n, n, false, false,
                   (res, v, α, β) -> begin
                       @. state = v + v0
                       compute_roundtrip!(s, tmp_state)
                       if iszero(β)
                           @. res = α * (v - v0 - r + state)
                       else
                           @. res = α * (v - v0 - r + state) + β * res
                       end
                       res
                   end; S)
end

function FixedPointSolver(s::BidirectionalSystem, Workspace; kwargs...)
    tmp_state = similar(s.fp_state)
    r = similar(s.fp_state)
    v = getdata(s.fp_state)
    n = length(v)
    S = typeof(v)
    ws = Workspace(n, n, S; kwargs...)
    BidirectionalSolver(s, ws, tmp_state, r)
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

function fp_solve!(solver::BidirectionalSolver;
                   return_stats = false, precondition = false,
                   spectral_projection = false, kwargs...)
    s = solver.system
    copyto!(solver.r, s.fp_state)
    compute_roundtrip!(s, solver.r; spectral_projection)
    r = getdata(solver.r)
    v0 = getdata(s.fp_state)
    @. r = r - v0
    op = compute_linear_operator(s, solver.tmp_state, v0, r; spectral_projection)
    op_pre = precondition ? compute_preconditioner(s, solver.tmp_state, v0, solver.r) : I
    krylov_solve!(solver, op; N = op_pre, kwargs...)
    res_state = Krylov.solution(solver.workspace)
    δ = getdata(res_state)
    δ = δ isa Tuple ? first(δ) : δ
    @. v0 = v0 + δ
    copyto!(getdata(solver.tmp_state), v0)
    compute_roundtrip!(s, solver.tmp_state; spectral_projection)
    res = (reflected = get_source(s.start_source),
           transmitted = get_source(s.end_source))
    if return_stats
        stats = solver.workspace.stats
        (res, stats)
    else
        res
    end
end

function fp_solve!(s::BidirectionalSystem; itmax = 100, spectral_projection = false)
    fp = get_fp_state(s)
    for i in 1:itmax
        compute_roundtrip!(s, fp; spectral_projection)
    end
    set_fp_state!(s, fp)
    (reflected = get_source(s.start_source),
     transmitted = get_source(s.end_source))
end

function test_adjoint(solver, rand!)
    s = solver.system
    
    copyto!(solver.r, s.fp_state)
    compute_roundtrip!(s, solver.r)
    r = getdata(solver.r)
    v0 = getdata(s.fp_state)
    @. r = r - v0
    op = compute_linear_operator(s, solver.tmp_state, v0, r)
    
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
