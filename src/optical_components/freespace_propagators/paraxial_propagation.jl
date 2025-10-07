# Edward A. Sziklas and A. E. Siegman, "Mode calculations in unstable
# resonators with flowing saturable gain. 2: Fast Fourier transform
# method," Appl. Opt. 14, 1874-1889 (1975)
# https://doi.org/10.1364/AO.14.001874

# Careful that z != 0
"""
    ParaxialProp(u::ScalarField, ds::NTuple, ds′::NTuple, z::Real; use_cache=true, filter=nothing, double_precision_kernel=use_cache)
    ParaxialProp(u::ScalarField, ds′::NTuple, z::Real; kwargs...)
    ParaxialProp(u::ScalarField, z::Real; kwargs...)

Paraxial propagation with optional grid resampling.

Convenience wrapper: uses `ASProp` (paraxial mode) when no magnification, or
`CollinsProp` when grid resampling is needed.

# Arguments
- `u::ScalarField`: Field template
- `ds::NTuple`: Input sampling (defaults to `u.ds`)
- `ds′::NTuple`: Output sampling (triggers Collins if ≠ input)
- `z::Real`: Propagation distance
- `use_cache::Bool`: Cache kernels (default: true)
- `filter`: Optional frequency filter
- `double_precision_kernel::Bool`: Use Float64 kernels (default: use_cache)

# Examples
```julia
xv, yv = spatial_vectors(256, 256, 0.25, 0.25)

u = ScalarField(Gaussian(5.0)(xv, yv), (0.5, 0.5), 1.064)

# No magnification -> uses ASProp
prop1 = ParaxialProp(u, 1000.0)

# With magnification -> uses CollinsProp
prop2 = ParaxialProp(u, (1.0, 1.0), 1000.0)
```

See also: [`ASProp`](@ref), [`CollinsProp`](@ref)
"""
function ParaxialProp(u::ScalarField{U, Nd},
                      ds::NTuple{Nd, Real},
                      ds′::NTuple{Nd, Real},
                      z::Real;
                      use_cache::Bool = true,
                      filter = nothing,
                      double_precision_kernel::Bool = use_cache) where {Nd,
                                                                        U <:
                                                                        AbstractArray{<:Complex}}
    if ds == ds′
        ASProp(u, ds, z; use_cache, filter, paraxial = true, double_precision_kernel)
    else
        CollinsProp(u, ds, ds′, (1, z, 1); use_cache, double_precision_kernel)
    end
end

# Careful that z != 0
function ParaxialProp(u::ScalarField{U, Nd},
                      ds′::NTuple{Nd, Real},
                      z::Real;
                      use_cache::Bool = true,
                      filter = nothing,
                      double_precision_kernel::Bool = use_cache) where {Nd,
                                                                        U <:
                                                                        AbstractArray{<:Complex}}
    ParaxialProp(u, Tuple(u.ds), ds′, z; use_cache, double_precision_kernel)
end

function ParaxialProp(u::ScalarField,
                      z::Real;
                      use_cache::Bool = true,
                      filter = nothing,
                      double_precision_kernel::Bool = use_cache)
    ASProp(u, z; use_cache, filter, paraxial = true, double_precision_kernel)
end
