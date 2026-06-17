struct BidirectionalSystem{S1, S2, C, P, K}
    start_source::S1
    end_source::S2
    components::C
    fp_state::P
    fp_state_keys::K
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
    BidirectionalSystem(start_source, end_source, all_components, fp_state, state_keys)
end

function BidirectionalSystem(start_source::AbstractBidirectionalSource,
                             components::Vararg{AbstractBidirectionalComponent})
    end_source = zero(start_source; n0 = get_n0_right(Base.last(components)))
    BidirectionalSystem(start_source, end_source, components...)
end

get_fp_state(s::BidirectionalSystem) = copy(s.fp_state)

set_fp_state!(s::BidirectionalSystem, fp_state) = copyto!(s.fp_state, fp_state)

function compute_roundtrip!(s::BidirectionalSystem, fp_state::ComponentArray)
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

compute_roundtrip!(s::BidirectionalSystem) = compute_roundtrip!(s, s.fp_state)

function fp_solve!(s::BidirectionalSystem; tol=1e-8, maxiter=100)
    fp = get_fp_state(s)
    for i in 1:maxiter
        fp_old = copy(fp)
        compute_roundtrip!(s, fp)
        # fp_data = getdata(fp)
        # fp_old_data = getdata(fp_old)
        # norm(fp_data .- fp_old_data) / (norm(fp_old_data) + eps()) < tol && break
    end
    set_fp_state!(s, fp)
    (reflected = get_source(s.start_source),
     transmitted = get_source(s.end_source))
end
