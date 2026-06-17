struct BidirectionalSystem{S1, S2, C, P, V}
    start_source::S1
    end_source::S2
    components::C
    fp_state::P
    fp_state_views::V
end

Functors.@functor BidirectionalSystem (components,)

_interleave(ifaces::Tuple{T}, ::Tuple{}) where {T} = ifaces
_interleave(ifaces, comps) =
    (first(ifaces), first(comps), _interleave(Base.tail(ifaces), Base.tail(comps))...)

coalesce_state(s, _) = s
coalesce_state(::Nothing, null_array) = null_array

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
    null_array = similar(U, 1)(undef, 0)
    states = map(c -> coalesce_state(initial_state(c), null_array), all_components)
    fp_state = ComponentArray(NamedTuple{state_keys}(states))
    fp_state_views = map(k -> getproperty(fp_state, k), state_keys)
    BidirectionalSystem(start_source, end_source, all_components, fp_state, fp_state_views)
end

