"""
    generate_speckle(ns, ds, λ, NA; envelope=nothing, center=(0,0), t=Id2D(), normalize=true)

Generate random speckle pattern with controlled numerical aperture.

Creates a complex random field with Fourier components limited by the numerical
aperture, producing realistic speckle with controlled correlation length.

# Arguments
- `ns`: Grid dimensions - (nx,) for 1D, (nx, ny) for 2D, (nx, ny, nz) for 3D
- `ds`: Pixel sizes - (dx,), (dx, dy), or (dx, dy, dz)
- `λ`: Wavelength
- `NA`: Numerical aperture controlling speckle size
- `envelope=nothing`: Optional envelope function (Mode type)
- `center=(0,0)`: Center position for envelope
- `t=Id2D()`: Coordinate transformation for envelope
- `normalize=true`: Normalize the speckle distribution to unit power

# Returns
Complex array with random speckle pattern.

# Examples
```jldoctest
julia> speckle = generate_speckle((64, 64), (1.0, 1.0), 1.064, 0.1);

julia> isapprox(sum(abs2, speckle), 1)
true

julia> envelope = Gaussian(20.0);

julia> speckle_env = generate_speckle((64, 64), (1.0, 1.0), 1.064, 0.1; envelope=envelope);

julia> isapprox(sum(abs2, speckle_env), 1)
true

julia> speckle_1d = generate_speckle((128,), (0.5,), 1.064, 0.2);

julia> isapprox(sum(abs2, speckle_1d) * 0.5, 1)
true
```
"""
function generate_speckle(ns::NTuple{N, Integer},
                          ds::NTuple{Nd, Real},
                          λ::Real,
                          NA::Real;
                          envelope::Union{Nothing, Mode{Nd}} = nothing,
                          center::NTuple{Nd, Real} = ntuple(_ -> 0, Nd),
                          t::AbstractAffineMap = Id2D(),
                          normalize = true) where {N, Nd}
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
    if normalize
        speckle ./= (norm(speckle) * sqrt(prod(ds)))
    end
    speckle
end
