struct BidirectionalSystem{S1, S2, C, P, K, J}
    start_source::S1
    end_source::S2
    start_source_adj::S1
    end_source_adj::S2
    components::C
    fp_state::P
    fp_state_keys::K
    spectral_projector::J
end

Functors.@functor BidirectionalSystem (components,)

_interleave(ifaces::Tuple{T}, ::Tuple{}) where {T} = ifaces
_interleave(ifaces, comps) =
    (first(ifaces), first(comps), _interleave(Base.tail(ifaces), Base.tail(comps))...)

coalesce_state(s) = fmap(x -> isnothing(x) ? [] : x, s)

function state_view(fp_state::ComponentArray, k::Symbol)
    v = getproperty(fp_state, k)
    if Functors.isleaf(v) && isempty(v)
        nothing
    else
        fmap(x -> isempty(x) ? nothing : x, NamedTuple(v))
    end
end

function BidirectionalSystem(start_source::AbstractBidirectionalSource{U},
                             end_source::AbstractBidirectionalSource{U},
                             components::Vararg{AbstractBidirectionalComponent}) where {U}
    start_source_adj = zero(start_source)
    end_source_adj = zero(end_source)
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
    BidirectionalSystem(start_source, end_source, start_source_adj, end_source_adj,
                        all_components, fp_state, state_keys, spectral_projector)
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

function compute_roundtrip!(s::BidirectionalSystem,
                            start_source::AbstractBidirectionalSource,
                            end_source::AbstractBidirectionalSource,
                            fp_state::ComponentArray;
                            spectral_projection::Bool = false)
    if spectral_projection
        apply_spectral_projection!(s, fp_state)
    end
    fp_state_views = @ignore_derivatives map(k -> state_view(fp_state, k), s.fp_state_keys)
    u = propagate(start_source)
    for (p, state) in zip(s.components, fp_state_views)
        u = propagate!(u, state, p)
    end
    uf = propagate!(u, end_source)
    u = propagate(end_source)
    for (p, state) in zip(reverse(s.components), reverse(fp_state_views))
        u = inverse_propagate!(u, state, p)
    end
    ur = inverse_propagate!(u, start_source)
    (uf, ur)
end

function compute_roundtrip_adjoint!(s::BidirectionalSystem,
                                    start_source_adj::AbstractBidirectionalSource,
                                    end_source_adj::AbstractBidirectionalSource,
                                    fp_state::ComponentArray;
                                    spectral_projection::Bool = false)
    fp_state_views = map(k -> state_view(fp_state, k), s.fp_state_keys)
    u = propagate(start_source_adj)
    u = inverse_propagate_adjoint!(u, start_source_adj)
    for (p, state) in zip(s.components, fp_state_views)
        u = inverse_propagate_adjoint!(u, state, p)
    end
    ∂uf = u
    u = propagate(end_source_adj)
    u = propagate_adjoint!(u, end_source_adj)
    for (p, state) in zip(reverse(s.components), reverse(fp_state_views))
        u = propagate_adjoint!(u, state, p)
    end
    ∂ur = u
    if spectral_projection
        apply_spectral_projection!(s, fp_state)
    end
    (∂uf, ∂ur)
end

struct BidirectionalSolver{W, P}
    workspace::W
    tmp_state::P
    r::P
end

function compute_linear_operator(s::BidirectionalSystem,
                                 s_in::AbstractBidirectionalSource,
                                 s_out::AbstractBidirectionalSource,
                                 s_in_adj::AbstractBidirectionalSource,
                                 s_out_adj::AbstractBidirectionalSource,
                                 tmp_state::ComponentArray, v0, r;
                                 adjoint::Bool = false,
                                 spectral_projection::Bool = false)
    state = getdata(tmp_state)
    T = eltype(state)
    n = length(state)
    S = typeof(state)
    s_in, s_out, s_in_adj, s_out_adj = adjoint ?
        (s_in_adj, s_out_adj, s_in, s_out) : (s_in, s_out, s_in_adj, s_out_adj)
    function prod!(res, v, α, β)
        @. state = v + v0
        compute_roundtrip!(s, s_in, s_out, tmp_state; spectral_projection)
        if iszero(β)
            @. res = α * (v + v0 + r - state)
        else
            @. res = α * (v + v0 + r - state) + β * res
        end
        res
    end
    function ctprod!(res, v, α, β)
        @. state = v
        compute_roundtrip_adjoint!(s, s_in_adj, s_out_adj,
                                   tmp_state; spectral_projection)
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
    tmp_state = similar(s.fp_state)
    r = similar(s.fp_state)
    v = getdata(r)
    n = length(v)
    S = typeof(v)
    ws = Workspace(n, n, S; kwargs...)
    BidirectionalSolver(ws, tmp_state, r)
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
    s_in, s_out = s.start_source, s.end_source
    s_in_adj, s_out_adj = zero(s_in), zero(s_out)
    copyto!(solver.r, s.fp_state)
    compute_roundtrip!(s, s_in, s_out, solver.r; spectral_projection)
    r = getdata(solver.r)
    v0 = getdata(s.fp_state)
    @. r = r - v0
    op = compute_linear_operator(s, s_in, s_out, s_in_adj, s_out_adj,
                                 solver.tmp_state, v0, r; spectral_projection)
    krylov_solve!(solver, op; kwargs...)
    res_state = Krylov.solution(solver.workspace)
    δ = getdata(res_state)
    δ = δ isa Tuple ? first(δ) : δ
    @. v0 = v0 + δ
    copyto!(solver.tmp_state, s.fp_state)
    uf, ur = compute_roundtrip!(s, s_in, s_out, solver.tmp_state; spectral_projection)
    (reflected = ur, transmitted = uf)
end

function fp_solve!(s::BidirectionalSystem; itmax = 100, spectral_projection = false)
    s_in, s_out = s.start_source, s.end_source
    s_in_adj, s_out_adj = zero(s_in), zero(s_out)
    for i in 1:itmax-1
        compute_roundtrip!(s, s_in, s_out, s.fp_state; spectral_projection)
    end
    uf, ur = compute_roundtrip!(s, s_in, s_out, s.fp_state; spectral_projection)
    (reflected = ur, transmitted = uf)
end

function test_adjoint(s, solver, rand!)
    s_in, s_out = s.start_source, s.end_source
    s_in_adj, s_out_adj = zero(s_in), zero(s_out)

    copyto!(solver.r, s.fp_state)
    compute_roundtrip!(s, s_in, s_out, solver.r)
    r = getdata(solver.r)
    v0 = getdata(s.fp_state)
    @. r = r - v0
    op = compute_linear_operator(s, s_in, s_out, s_in_adj, s_out_adj, solver.tmp_state, v0, r)
    
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
