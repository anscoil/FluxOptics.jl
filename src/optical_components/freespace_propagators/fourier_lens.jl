"""
    FourierLens(u::ScalarField, ds::NTuple, ds′::NTuple, fl::Real; use_cache=true, double_precision_kernel=use_cache)
    FourierLens(u::ScalarField, ds′::NTuple, fl::Real; kwargs...)

Ideal Fourier lens with grid resampling.

Performs Fourier transform with magnification and grid resampling. Equivalent to
`CollinsProp` with ABCD = (0, f, 0).  Unlike the FFT, it does not perform an fftshift,
so the beams shift by an amount related to their linear phase, as would be expected.

# Arguments
- `u::ScalarField`: Field template
- `ds::NTuple`: Input sampling (defaults to `u.ds`)
- `ds′::NTuple`: Output grid sampling
- `fl::Real`: Focal length
- `use_cache::Bool`: Cache kernels (default: true)
- `double_precision_kernel::Bool`: Use Float64 kernels (default: use_cache)

# Examples
```julia
xv, yv = spatial_vectors(256, 256, 0.25, 0.25)

u = ScalarField(Gaussian(5.0)(xv, yv), (0.5, 0.5), 1.064)

# Fourier plane with magnification
lens = FourierLens(u, (1.5, 1.5), 1000.0)

# Beam collimiation
u_col = propagate(u, lens, Forward)
```

See also: [`CollinsProp`](@ref), [`ParaxialProp`](@ref)
"""
function FourierLens(u::ScalarField{U, Nd},
                     ds::NTuple{Nd, Real},
                     ds′::NTuple{Nd, Real},
                     fl::Real;
                     use_cache::Bool = true,
                     double_precision_kernel::Bool = use_cache) where {Nd,
                                                                       U <:
                                                                       AbstractArray{<:Complex}}
    CollinsProp(u, ds, ds′, (0, fl, 0); use_cache, double_precision_kernel)
end

function FourierLens(u::ScalarField{U, Nd},
                     ds′::NTuple{Nd, Real},
                     fl::Real;
                     use_cache::Bool = true,
                     double_precision_kernel::Bool = use_cache) where {Nd,
                                                                       U <:
                                                                       AbstractArray{<:Complex}}
    CollinsProp(u, ds′, (0, fl, 0); use_cache, double_precision_kernel)
end
