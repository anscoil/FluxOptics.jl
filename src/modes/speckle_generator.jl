function generate_speckle(
        ns::NTuple{N, Integer}, ds::NTuple{Nd, Real}, λ::Real, NA::Real;
        envelope::Union{Nothing, Mode{Nd}} = nothing,
        center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd),
        t::AbstractAffineMap = Id2D()
) where {N, Nd}
    @assert Nd in 1:3
    @assert N >= Nd
    ns_d = ns[1:Nd]
    fs_axes = [fftfreq(nx, 1/dx) for (nx, dx) in zip(ns_d, ds)]
    r02 = (NA/λ)^2
    ball = x -> sum(abs2, x) <= r02 ? 1.0 : 0.0
    filter = [ball(x) for x in Iterators.product(fs_axes...)]
    speckle = cis.(2π*rand(ns...))
    speckle .*= filter
    ifft!(speckle, Tuple(1:Nd))
    if !isnothing(envelope)
        xs = spatial_vectors(ns_d, ds; offset = center)
        speckle .*= envelope(xs..., t)
    end
    speckle
end
