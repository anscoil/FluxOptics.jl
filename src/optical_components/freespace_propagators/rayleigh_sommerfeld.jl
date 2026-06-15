function rs_kernel(x::T, y::T, λ::T, n0::Tp, z::Tp, nrm_f::Tp) where {T <: Real, Tp <: Real}
    x, y = Tp(x), Tp(y)
    k = Tp(2π/λ)
    r = sqrt(x^2 + y^2 + z^2)
    kernel = z > 0 ? (cis(k*r)/r)*(z/r)*(1/r-im*k) :
             conj(cis(k*r)/r)*(-z/r)*(1/r+im*k)
    Complex{T}(kernel * nrm_f)
end

function rs_tilted_kernel(x::T, y::T, λ::T, θx::T, θy::T, track_tilts::Bool, n0::Tp, z::Tp,
                          nrm_f::Tp) where {T <: Real, Tp <: Real}
    x, y = Tp(x), Tp(y)
    f0x, f0y = sin(θx)/λ, sin(θy)/λ
    k = Tp(2π/λ)
    r = sqrt(x^2 + y^2 + z^2)
    lin_phase = track_tilts ? Complex{Tp}(1) : cis(-Tp(2)*π*(x*f0x+y*f0y))
    kernel = z > 0 ? (cis(k*r)/r)*lin_phase*(z/r)*(1/r-im*k) :
             conj(cis(k*r)/r)*lin_phase*(-z/r)*(1/r+im*k)
    Complex{T}(kernel * nrm_f)
end

function rs_valid_distance(nx, ny, dx, dy, λ)
    if dx < λ/2 && dy < λ/2
        return 0.0
    end
    zc_x = dx < λ/2 ? 0.0 : (nx*dx/2) * sqrt(4*dx^2/λ^2 - 1)
    zc_y = dy < λ/2 ? 0.0 : (ny*dy/2) * sqrt(4*dy^2/λ^2 - 1)
    max(zc_x, zc_y)
end

struct RSKernelProp{M, K, Tp} <: AbstractPropagator{M, K}
    trainability::Val{M}
    kernel::K
    track_tilts::Bool
    n0::Tp
    z::Tp
    nrm_f::Tp
end

Functors.@functor RSKernelProp ()

function RSKernelProp(u::ScalarField{U, Nd},
                      ds::NTuple{Nd, Real},
                      z::Real;
                      use_cache::Bool = true,
                      track_tilts::Bool = false,
                      n0::Real = 1,
                      double_precision_kernel::Bool = use_cache
                      ) where {T, U <: AbstractArray{Complex{T}}, Nd}
    ns = size(u)[1:Nd]
    ns′ = map(n -> 2*n-1, ns)
    cache_size = use_cache ? prod(size(u)[(Nd + 1):end]) : 0
    kernel = ConvolutionKernel(u.electric, ns, ds, cache_size; normalize = false)
    Tp = double_precision_kernel ? Float64 : T
    nrm_f = Tp(prod(ds)/2π/prod(ns′))
    RSKernelProp(Val(Static), kernel, track_tilts, Tp(n0), Tp(z), nrm_f)
end

get_kernels(p::RSKernelProp) = (p.kernel,)

function build_kernel_key_args(p::RSKernelProp, u::ScalarField)
    if is_on_axis(u)
        (select_lambdas(u),)
    else
        (select_lambdas(u), select_tilts(u)...)
    end
end

function build_kernel_args(p::RSKernelProp, u::ScalarField)
    if is_on_axis(u)
        (p.n0, p.z, p.nrm_f)
    else
        (p.track_tilts, p.n0, p.z, p.nrm_f)
    end
end

function _propagate_core!(apply_kernel_fns::F,
                          u::ScalarField,
                          p::RSKernelProp,
                          ::Type{<:Direction}) where {F}
    apply_kernel_fn!, = apply_kernel_fns
    if is_on_axis(u)
        apply_kernel_fn!(u.electric, rs_kernel)
    else
        apply_kernel_fn!(u.electric, rs_tilted_kernel)
    end
    u
end

"""
    RSProp(u::ScalarField, z::Real; use_cache=true, track_tilts=false, double_precision_kernel=use_cache)
    RSProp(u::ScalarField, ds::NTuple, z::Real; kwargs...)

Rayleigh-Sommerfeld diffraction propagation.

Uses Rayleigh-Sommerfeld diffraction integral for field propagation.
Prevents aliasing for large propagation distances but requires finer sampling for short distances.

# Arguments
- `u::ScalarField`: Field template
- `z::Real`: Propagation distance
- `ds::NTuple`: Custom spatial sampling (defaults to `u.ds`)
- `use_cache::Bool`: Cache kernels (default: true)
- `n0::Real`: Refractive index (default: 1)
- `track_tilts::Bool`: Track tilt evolution (default: false)
- `double_precision_kernel::Bool`: Use Float64 kernels (default: use_cache)

# Validity
Critical distance:
- `z_c = (N dx / 2) √(4dx²/λ² - 1)`

If `z < z_c`, a warning is issued. Use `ASProp` or finer sampling instead.

# Examples
```julia
u = ScalarField(ones(ComplexF64, 256, 256), (0.5, 0.5), 1.064)  # dx < λ/2

# Short distance propagation
prop = RSProp(u, 100.0)

u_out = propagate(u, prop)
```

See also: [`ASProp`](@ref)
"""
struct RSProp{M, C} <: AbstractSequence{M}
    trainability::Val{M}
    optical_components::C
end

Functors.@functor RSProp (optical_components,)

function RSProp(u::ScalarField{U, Nd},
                ds::NTuple{Nd, Real},
                z::Real;
                use_cache::Bool = true,
                track_tilts::Bool = false,
                n0::Real = 1,
                double_precision_kernel::Bool = use_cache
                ) where {T, U <: AbstractArray{Complex{T}}, Nd}
    ns = size(u)[1:Nd]
    zc = rs_valid_distance(ns..., ds..., minimum(u.lambdas.collection)/n0)
    if abs(z) < zc
        @warn """RSProp: propagation distance z=$z is below critical distance zc=$zc.
                   Numerical artifacts expected. Consider using ASProp or finer sampling (dx < λ/2)."""
    end
    rs = RSKernelProp(u, ds, z; use_cache, track_tilts, n0, double_precision_kernel)
    wrapper = FourierWrapper(rs.kernel.p_f, rs.kernel.nrm_f, rs)
    pad_op = PadCropOperator(u, rs.kernel.u_plan; store_ref = true)
    crop_op = adjoint(pad_op)
    optical_components = (pad_op, get_sequence(wrapper)..., crop_op)
    M = get_trainability(wrapper)
    RSProp(Val(M), optical_components)
end

function RSProp(u::ScalarField,
                z::Real;
                use_cache::Bool = true,
                track_tilts::Bool = false,
                n0::Real = 1,
                double_precision_kernel::Bool = use_cache)
    RSProp(u, Tuple(u.ds), z; use_cache, track_tilts, n0, double_precision_kernel)
end

get_sequence(p::RSProp) = p.optical_components
