function as_kernel(fx::T, fy::T, λ::T, n0::Tp, z::Tp, filter::H,
                   z_pos::Val{true}) where {T <: Real, Tp <: Real, H}
    fx, fy, λ = Tp(fx), Tp(fy), Tp(λ)/n0
    f² = complex(inv(λ^2))
    v = isnothing(filter) ? Complex{Tp}(1) : Complex{Tp}(filter(fx, fy))
    Complex{T}(cis(Tp(2)*π*z*sqrt(f² - fx^2 - fy^2)) * v)
end

function as_kernel(fx::T, fy::T, λ::T, n0::Tp, z::Tp, filter::H,
                   z_pos::Val{false}) where {T <: Real, Tp <: Real, H}
    conj(as_kernel(fx, fy, λ, n0, -z, filter, Val(true)))
end

function as_kernel(fx::T, λ::T, n0::Tp, z::Tp, filter::H,
                   z_pos::Val{true}) where {T <: Real, Tp <: Real, H}
    fx, λ = Tp(fx), Tp(λ)/n0
    f² = complex(inv(λ)^2)
    v = isnothing(filter) ? Complex{Tp}(1) : Complex{Tp}(filter(fx))
    Complex{T}(cis(Tp(2)*π*z*sqrt(f² - fx^2)) * v)
end

function as_kernel(fx::T, λ::T, n0::Tp, z::Tp, filter::H,
                   z_pos::Val{false}) where {T <: Real, Tp <: Real, H}
    conj(as_kernel(fx, λ, n0, -z, filter, Val(true)))
end

function as_paraxial_kernel(fx::T, fy::T, λ::T, θx::T, θy::T, track_tilts::Bool,
                            n0::Tp, z::Tp, filter::H) where {T <: Real, Tp <: Real, H}
    fx, fy, λ = Tp(fx), Tp(fy), Tp(λ/n0)
    f0x, f0y = sin(θx)/λ, sin(θy)/λ
    offset = track_tilts ? 2*(f0x*fx + f0y*fy) : Tp(0)
    v = isnothing(filter) ? Complex{Tp}(1) : Complex{Tp}(filter(fx, fy))
    Complex{T}(cis(-π*λ*z*((fx+f0x)^2 + (fy+f0y)^2) - offset + Tp(2)*π*z/λ) * v)
end

function as_paraxial_kernel(fx::T, λ::T, θx::T, track_tilts::Bool, n0::Tp, z::Tp,
                            filter::H) where {T <: Real, Tp <: Real, H}
    fx, λ = Tp(fx), Tp(λ/n0)
    f0x = sin(θx)/λ
    offset = track_tilts ? 2*f0x*fx : Tp(0)
    v = isnothing(filter) ? Complex{Tp}(1) : Complex{Tp}(filter(fx))
    Complex{T}(cis(-π*λ*z*(fx+f0x)^2 - offset + Tp(2)*π*z/λ) * v)
end

function tilted_as_kernel(fx::T, fy::T, λ::T, θx::T, θy::T, track_tilts::Bool,
                          n0::Tp, z::Tp, filter::H,
                          z_pos::Val{true}) where {T <: Real, Tp <: Real, H}
    fx, fy, λ = Tp(fx), Tp(fy), Tp(λ)/n0
    θx, θy = Tp(θx), Tp(θy)
    f² = complex(inv(λ)^2)
    f0x, f0y = sin(θx)/λ, sin(θy)/λ
    offset = track_tilts ? (f0x*fx + f0y*fy)*λ : Tp(0)
    v = isnothing(filter) ? Complex{Tp}(1) : Complex{Tp}(filter(fx+f0x, fy+f0y))
    Complex{T}(cis(Tp(2)*π*z*(sqrt(f²-(fx+f0x)^2-(fy+f0y)^2) + offset)) * v)
end

function tilted_as_kernel(fx::T, fy::T, λ::T, θx::T, θy::T, track_tilts::Bool,
                          n0::Tp, z::Tp, filter::H,
                          z_pos::Val{false}) where {T <: Real, Tp <: Real, H}
    conj(tilted_as_kernel(fx, fy, λ, θx, θy, track_tilts, n0, -z, filter, Val(true)))
end

function tilted_as_kernel(fx::T, λ::T, θx::T, track_tilts::Bool, n0::Tp, z::Tp, filter::H,
                          z_pos::Val{true}) where {T <: Real, Tp <: Real, H}
    fx, λ, θx = Tp(fx), Tp(λ)/n0, Tp(θx)
    f² = complex(inv(λ^2))
    f0x = sin(θx)/λ
    offset = track_tilts ? f0x*fx*λ : Tp(0)
    v = isnothing(filter) ? Complex{Tp}(1) : Complex{Tp}(filter(fx+f0x))
    Complex{T}(cis(Tp(2)*π*z*(sqrt(f²-(fx+f0x)^2) + offset)) * v)
end

function tilted_as_kernel(fx::T, λ::T, θx::T, track_tilts::Bool, n0::Tp, z::Tp, filter::H,
                          z_pos::Val{false}) where {T <: Real, Tp <: Real, H}
    conj(tilted_as_kernel(fx, λ, θx, track_tilts, n0, -z, filter, Val(true)))
end

struct ASKernelProp{M, K, T, Tp, H} <: AbstractPropagator{M, K, T}
    kernel::K
    track_tilts::Bool
    n0::Tp
    z::Tp
    filter::H
    is_paraxial::Bool

    function ASKernelProp(u::ScalarField{U, Nd},
                          ds::NTuple{Nd, Real},
                          z::Real;
                          use_cache::Bool = true,
                          track_tilts::Bool = false,
                          n0::Real = 1,
                          filter::H = nothing,
                          paraxial::Bool = false,
                          double_precision_kernel::Bool
                          = use_cache) where {Nd, T, H, U <: AbstractArray{Complex{T}}}
        ns = size(u)[1:Nd]
        cache_size = use_cache ? prod(size(u)[(Nd + 1):end]) : 0
        kernel = FourierKernel(u.electric, ns, ds, cache_size)
        Tp = double_precision_kernel ? Float64 : T
        new{Static, typeof(kernel), T, Tp, H}(kernel, track_tilts, Tp(n0), Tp(z), filter,
                                              paraxial)
    end
end

Functors.@functor ASKernelProp ()

get_data(p::ASKernelProp) = p.kernel

get_kernels(p::ASKernelProp) = (p.kernel,)

function build_kernel_key_args(p::ASKernelProp, u::ScalarField)
    if is_on_axis(u) && !p.is_paraxial
        (select_lambdas(u),)
    else
        (select_lambdas(u), select_tilts(u)...)
    end
end

function build_kernel_args(p::ASKernelProp, u::ScalarField)
    if p.is_paraxial
        (p.track_tilts, p.n0, p.z, p.filter)
    else
        if is_on_axis(u)
            (p.n0, p.z, p.filter, Val(sign(p.z) > 0))
        else
            (p.track_tilts, p.n0, p.z, p.filter, Val(sign(p.z) > 0))
        end
    end
end

function _propagate_core!(apply_kernel_fns::F,
                          u::ScalarField,
                          p::ASKernelProp,
                          ::Type{<:Direction}) where {F}
    apply_kernel_fn!, = apply_kernel_fns
    if p.is_paraxial
        apply_kernel_fn!(u.electric, as_paraxial_kernel)
    else
        if is_on_axis(u)
            apply_kernel_fn!(u.electric, as_kernel)
        else
            apply_kernel_fn!(u.electric, tilted_as_kernel)
        end
    end
    u
end

"""
    ASProp(u::ScalarField, ds::NTuple, z::Real; use_cache=true, n0=1, filter=nothing, paraxial=false, double_precision_kernel=use_cache)
    ASProp(u::ScalarField, z::Real; kwargs...)

Angular Spectrum propagation in homogeneous media.

Propagates scalar fields through uniform medium with refractive index `n0` using FFT-based Angular Spectrum method.
Automatically handles both on-axis and tilted beams (unified implementation).

# Arguments
- `u::ScalarField`: Field template
- `ds::NTuple`: Custom spatial sampling (defaults to `u.ds`)
- `z::Real`: Propagation distance (positive or negative)
- `use_cache::Bool`: Cache kernels (default: true)
- `n0::Real`: Refractive index (default: 1)
- `filter`: Optional frequency filter function `(fx, fy) -> transmission`
- `paraxial::Bool`: Use paraxial approximation (default: false)
- `double_precision_kernel::Bool`: Compute kernels in Float64 (default: use_cache)

# Returns
Static propagator component (not trainable).

# Notes
When `use_cache=true`, kernels are computed once and stored, so using Float64 precision
(`double_precision_kernel=true`) adds no computational cost while improving accuracy.
For ComplexF64 fields, kernels are always computed in Float64 regardless of this setting.
The option primarily affects ComplexF32 fields (e.g., CUDA applications).

# Examples
```julia
u = ScalarField(ones(ComplexF64, 256, 256), (2.0, 2.0), 1.064)

# Simple propagation in air
prop = ASProp(u, 1000.0)

# In glass
prop_glass = ASProp(u, 1000.0; n0=1.5)

# With low-pass filter
filter_lp = (fx, fy) -> sqrt(fx^2 + fy^2) < 0.3 ? 1.0 : 0.0
prop_filtered = ASProp(u, 1000.0; filter=filter_lp)
```

See also: [`ASPropZ`](@ref), [`ParaxialProp`](@ref), [`RSProp`](@ref)
"""
struct ASProp{M, C} <: AbstractSequence{M}
    optical_components::C

    function ASProp(optical_components::C) where {N,
                                                  C <: NTuple{N, AbstractPipeComponent}}
        new{Trainable, C}(optical_components)
    end

    function ASProp(u::ScalarField{U, Nd},
                    ds::NTuple{Nd, Real},
                    z::Real;
                    use_cache::Bool = true,
                    track_tilts::Bool = false,
                    n0::Real = 1,
                    filter = nothing,
                    paraxial::Bool = false,
                    double_precision_kernel::Bool = use_cache) where {U, Nd}
        kernel = ASKernelProp(u, ds, z; use_cache, track_tilts, n0, filter, paraxial,
                              double_precision_kernel)
        wrapper = FourierWrapper(kernel.kernel.p_f, kernel)
        M = get_trainability(wrapper)
        optical_components = get_sequence(wrapper)
        C = typeof(optical_components)
        new{M, C}(optical_components)
    end

    function ASProp(u::ScalarField,
                    z::Real;
                    use_cache::Bool = true,
                    track_tilts::Bool = false,
                    n0::Real = 1,
                    filter = nothing,
                    paraxial::Bool = false,
                    double_precision_kernel::Bool = use_cache)
        ASProp(u, Tuple(u.ds), z; use_cache, track_tilts, n0, filter, paraxial,
               double_precision_kernel)
    end
end

Functors.@functor ASProp (optical_components,)

get_sequence(p::ASProp) = p.optical_components

"""
    ASPropZ(u::ScalarField, ds::NTuple, z::Real; n0=1, trainable=false, paraxial=false, filter=nothing, double_precision_kernel=false)
    ASPropZ(u::ScalarField, z::Real; kwargs...)

Angular Spectrum propagation with trainable distance.

Similar to `ASProp` but with trainable propagation distance `z` for optimization.
Uses automatic differentiation (Zygote, Enzyme) for gradients.

# Arguments
- `u::ScalarField`: Field template
- `ds::NTuple`: Custom spatial sampling (defaults to `u.ds`)
- `z::Real`: Initial propagation distance
- `n0::Real`: Refractive index (default: 1)
- `trainable::Bool`: Enable distance optimization (default: false)
- `paraxial::Bool`: Use paraxial approximation (default: false)
- `filter`: Optional frequency filter
- `double_precision_kernel::Bool`: Use Float64 kernels (default: false)

# Notes
Unlike ASProp kernels are never cached, so using Float64 precision
(`double_precision_kernel=true`) adds computational cost in case of ComplexF32 fields (e.g., CUDA applications).

# Examples
```julia
u = ScalarField(ones(ComplexF64, 256, 256), (2.0, 2.0), 1.064)

# Trainable distance
prop_z = ASPropZ(u, 500.0; trainable=true)

system = source |> phase |> prop_z
# ... optimize distance via gradients ...
```

**Note:** Slower than `ASProp` for fixed distances due to data allocation and AD overhead.

See also: [`ASProp`](@ref), [`Trainability`](@ref)
"""
struct ASPropZ{M, T, A, V, H} <: AbstractPureComponent{M}
    n0::T
    z::A
    track_tilts::Bool
    is_paraxial::Bool
    f_vec::V
    filter::H

    function ASPropZ(n0::T, z::A, is_paraxial::Bool, f_vec::V, filter::H) where {T, A, V, H}
        new{Trainable, T, A, V, H}(z, is_paraxial, f_vec, filter)
    end

    function ASPropZ(u::ScalarField{U, Nd},
                     ds::NTuple{Nd, Real},
                     z::Real;
                     n0::Real = 1,
                     trainable::Bool = false,
                     track_tilts::Bool = false,
                     paraxial::Bool = false,
                     filter::H = nothing,
                     double_precision_kernel::Bool = false) where {Nd, T, H,
                                                                   U <:
                                                                   AbstractArray{Complex{T}}}
        ns = size(u)[1:Nd]
        F = similar(U, real, 1)
        fs = [fftfreq(nx, 1/dx) |> F for (nx, dx) in zip(ns, ds)]
        f_vec = Nd == 2 ? (; x = fs[1], y = fs[2]') : (; x = fs[1])
        V = typeof(f_vec)
        M = trainable ? Trainable : Static
        Tp = double_precision_kernel ? Float64 : T
        z_arr = Tp.([z] |> F)
        new{M, Tp, typeof(z_arr), V, H}(Tp(n0), z_arr, track_tilts, paraxial, f_vec, filter)
    end

    function ASPropZ(u::ScalarField,
                     z::Real;
                     n0::Real = 1,
                     trainable::Bool = false,
                     track_tilts::Bool = false,
                     paraxial::Bool = false,
                     filter = nothing,
                     double_precision_kernel::Bool = false)
        ASPropZ(u, Tuple(u.ds), z; n0, trainable, track_tilts, paraxial, filter,
                double_precision_kernel)
    end
end

Functors.@functor ASPropZ (z,)

trainable(p::ASPropZ{<:Trainable}) = (; z = p.z)

function propagate(u::ScalarField, p::ASPropZ, direction::Type{<:Direction})
    ndims = length(p.f_vec)
    dims = ntuple(i -> i, ndims)
    lambdas = get_lambdas(u)
    if p.is_paraxial
        kernel = @. as_paraxial_kernel(p.f_vec..., lambdas, u.tilts.val..., p.track_tilts,
                                       p.n0, p.z, p.filter)
    else
        if is_on_axis(u)
            z_pos = Val(all(sign.(p.z) .> 0))
            kernel = @. as_kernel(p.f_vec..., lambdas, p.n0, p.z, p.filter, z_pos)
        else
            z_pos = Val(all(sign.(p.z) .> 0))
            kernel = @. tilted_as_kernel(p.f_vec..., lambdas, u.tilts.val..., p.track_tilts,
                                         p.n0, p.z, p.filter, z_pos)
        end
    end
    data = ifft(fft(u.electric, dims) .* conj_direction(kernel, direction), dims)
    set_field_data(u, data)
end
