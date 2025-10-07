function shift_kernel(fx::T, fy::T, θx::T, θy::T, z::Tp) where {T <: Real, Tp <: Real}
    fx, fy = Tp(fx), Tp(fy)
    θx, θy = Tp(θx), Tp(θy)
    f0x, f0y = tan(θx), tan(θy)
    Complex{T}(cis(-2π*z*(f0x*fx + f0y*fy)))
end

function shift_kernel(fx::T, θx::T, z::Tp) where {T <: Real, Tp <: Real}
    fx, θx = Tp(fx), Tp(θx)
    f0x = tan(θx)
    Complex{T}(cis(-2π*z*f0x*fx))
end

struct ShiftKernel{M, K, T, Tp} <: AbstractPropagator{M, K, T}
    kernel::K
    z::Tp

    function ShiftKernel(u::ScalarField{U, Nd},
                         ds::NTuple{Nd, Real},
                         z::Real;
                         use_cache::Bool = true,
                         double_precision_kernel::Bool = use_cache) where {Nd, T,
                                                                           U <:
                                                                           AbstractArray{Complex{T}}}
        ns = size(u)[1:Nd]
        cache_size = use_cache ? prod(size(u)[(Nd + 1):end]) : 0
        kernel = FourierKernel(u.electric, ns, ds, cache_size)
        Tp = double_precision_kernel ? Float64 : T
        new{Static, typeof(kernel), T, Tp}(kernel, Tp(z))
    end
end

Functors.@functor ShiftKernel ()

get_kernels(p::ShiftKernel) = (p.kernel,)

build_kernel_key_args(p::ShiftKernel, u::ScalarField) = (select_tilts(u)...,)

build_kernel_args(p::ShiftKernel, ::ScalarField) = (p.z,)

function _propagate_core!(apply_kernel_fns::F,
                          u::ScalarField,
                          p::ShiftKernel,
                          ::Type{<:Direction}) where {F}
    apply_kernel_fn!, = apply_kernel_fns
    apply_kernel_fn!(u.electric, shift_kernel)
    u
end

"""
    ShiftProp(u::ScalarField, ds::NTuple, z::Real; use_cache=true, double_precision_kernel=use_cache)
    ShiftProp(u::ScalarField, z::Real; kwargs...)

Geometric shift based on field tilts (no diffraction).

Translates field geometrically based on tilt metadata. Ignores diffraction entirely,
approximating ray optics. Useful for comparison with diffraction-based methods.

# Arguments
- `u::ScalarField`: Field template (must have tilts defined)
- `ds::NTuple`: Custom sampling (defaults to `u.ds`)
- `z::Real`: Propagation distance
- `use_cache::Bool`: Cache kernels (default: true)
- `double_precision_kernel::Bool`: Use Float64 kernels (default: use_cache)

# Physics

Shift: `Δx = z tan(θx)`, `Δy = z tan(θy)`

Applied in Fourier domain as linear phase: `exp(-i 2π z tan(θ) fx)`

**Limitation:** Only uses tilt metadata, cannot detect phase gradients in the
complex field itself.

# Examples
```julia
# Requires tilted field
xv, yv = spatial_vectors(256, 256, 0.25, 0.25)

u = ScalarField(Gaussian(50.0)(xv, yv), (2.0, 2.0), 1.064; tilts=(0.01, 0.0))

# Pure geometric shift
shift = ShiftProp(u, 1000.0)

# Compare with diffraction
as_prop = ASProp(u, 1000.0)
```

See also: [`ASProp`](@ref), [`Shift_BPM`](@ref)
"""
struct ShiftProp{M, C} <: AbstractSequence{M}
    optical_components::C

    function ShiftProp(u::ScalarField{U, Nd},
                       ds::NTuple{Nd, Real},
                       z::Real;
                       use_cache::Bool = true,
                       double_precision_kernel::Bool = use_cache) where {U, Nd}
        kernel = ShiftKernel(u, ds, z; use_cache, double_precision_kernel)
        wrapper = FourierWrapper(kernel.kernel.p_f, kernel)
        M = get_trainability(wrapper)
        optical_components = get_sequence(wrapper)
        C = typeof(optical_components)
        new{M, C}(optical_components)
    end

    function ShiftProp(u::ScalarField,
                       z::Real;
                       use_cache::Bool = true,
                       double_precision_kernel::Bool = use_cache)
        ShiftProp(u, Tuple(u.ds), z; use_cache, double_precision_kernel)
    end
end

Functors.@functor ShiftProp (optical_components,)

get_sequence(p::ShiftProp) = p.optical_components
