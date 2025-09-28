function as_kernel(fx::T, fy::T, λ::T, n0::Tp, z::Tp, filter::H,
                   z_pos::Val{true}) where {T <: Real, Tp <: Real, H}
    fx, fy, λ = Tp(fx), Tp(fy), Tp(λ)/n0
    f² = complex(inv(λ^2))
    v = isnothing(filter) ? Complex{Tp}(1) : Complex{Tp}(filter(fx, fy))
    Complex{T}(cis(Tp(2)*π*z*sqrt(f² - fx^2 - fy^2)) * v)
end

function as_kernel(fx::T, fy::T, λ::T, n0::Tp, z::Tp, filter::H,
                   z_pos::Val{false}) where {T <: Real, Tp <: Real, H}
    fx, fy, λ = Tp(fx), Tp(fy), Tp(λ)/n0
    f² = complex(inv(λ^2))
    v = isnothing(filter) ? Complex{Tp}(1) : Complex{Tp}(filter(fx, fy))
    Complex{T}(conj(cis(Tp(2)*π*(-z)*sqrt(f² - fx^2 - fy^2)) * v))
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
    fx, λ = Tp(fx), Tp(λ)/n0
    f² = complex(inv(λ)^2)
    v = isnothing(filter) ? Complex{Tp}(1) : Complex{Tp}(filter(fx))
    Complex{T}(conj(cis(Tp(2)*π*(-z)*sqrt(f² - fx^2)) * v))
end

function as_paraxial_kernel(fx::T, fy::T, λ::T, n0::Tp, z::Tp,
                            filter::H) where {T <: Real, Tp <: Real, H}
    fx, fy, λ = Tp(fx), Tp(fy), Tp(λ/n0)
    v = isnothing(filter) ? Complex{Tp}(1) : Complex{Tp}(filter(fx, fy))
    Complex{T}(cis(-π*λ*z*(fx^2 + fy^2)) * v)
end

function as_paraxial_kernel(fx::T, λ::T, n0::Tp, z::Tp,
                            filter::H) where {T <: Real, Tp <: Real, H}
    fx, λ = Tp(fx), Tp(λ/n0)
    v = isnothing(filter) ? Complex{Tp}(1) : Complex{Tp}(filter(fx))
    Complex{T}(cis(-π*λ*z*fx^2) * v)
end

struct ASKernel{M, K, T, Tp, H} <: AbstractPropagator{M, K, T}
    kernel::K
    is_paraxial::Bool
    n0::Tp
    z::Tp
    filter::H

    function ASKernel(u::ScalarField{U, Nd},
                      ds::NTuple{Nd, Real},
                      z::Real;
                      use_cache::Bool = true,
                      n0::Real = 1,
                      filter::H = nothing,
                      paraxial::Bool = false,
                      double_precision_kernel::Bool = use_cache) where {Nd, T, H,
                                                                        U <:
                                                                        AbstractArray{Complex{T}}}
        ns = size(u)[1:Nd]
        cache_size = use_cache ? prod(size(u)[(Nd + 1):end]) : 0
        kernel = FourierKernel(u.electric, ns, ds, cache_size)
        Tp = double_precision_kernel ? Float64 : T
        new{Static, typeof(kernel), T, Tp, H}(kernel, paraxial, Tp(n0), Tp(z), filter)
    end
end

Functors.@functor ASKernel ()

get_data(p::ASKernel) = p.kernel

get_kernels(p::ASKernel) = (p.kernel,)

build_kernel_key_args(p::ASKernel, u::ScalarField) = (select_lambdas(u),)

function build_kernel_args(p::ASKernel)
    if p.is_paraxial
        (p.n0, p.z, p.filter)
    else
        (p.n0, p.z, p.filter, Val(sign(p.z) > 0))
    end
end

function _propagate_core!(apply_kernel_fns::F,
                          u::AbstractArray,
                          p::ASKernel,
                          ::Type{<:Direction}) where {F}
    apply_kernel_fn!, = apply_kernel_fns
    if p.is_paraxial
        apply_kernel_fn!(u, as_paraxial_kernel)
    else
        apply_kernel_fn!(u, as_kernel)
    end
    u
end

"""
    ASProp(u::ScalarField, z::Real; use_cache=true, n0=1, filter=nothing, paraxial=false, double_precision_kernel=use_cache)
    ASProp(u::ScalarField, ds::NTuple, z::Real; kwargs...)

Angular Spectrum propagation method for scalar field propagation in uniform media.

Implements the angular spectrum method using FFT-based convolution for scalar optical
fields in homogeneous media with refractive index `n0`. This method provides a good 
balance between accuracy and computational efficiency, handling both paraxial and 
non-paraxial regimes with excellent performance for most practical optical propagation problems.

# Arguments
- `u::ScalarField`: Reference field for grid definition
- `z::Real`: Propagation distance (positive or negative)
- `ds::NTuple`: Optional custom spatial sampling intervals (defaults to `u.ds`)
- `use_cache=true`: Cache kernels for repeated propagations
- `n0=1`: Refractive index of medium  
- `filter=nothing`: Optional spatial frequency filter function
- `paraxial=false`: Use paraxial approximation
- `double_precision_kernel=use_cache`: Use Float64 for kernel computation (when caching, costs nothing and improves precision; only relevant for ComplexF32 fields)

# Returns
Static propagator component (not trainable).

# Notes
When `use_cache=true`, kernels are computed once and stored, so using Float64 precision
(`double_precision_kernel=true`) adds no computational cost while improving accuracy.
For ComplexF64 fields, kernels are always computed in Float64 regardless of this setting.
The option primarily affects ComplexF32 fields (e.g., CUDA applications).

# Examples
```jldoctest
julia> u = ScalarField(ones(ComplexF64, 64, 64), (1.0, 1.0), 1.064);

julia> prop = ASProp(u, 1000.0);  # 1 mm propagation in air

julia> prop_glass = ASProp(u, 1000.0; n0=1.5);  # 1 mm in glass

julia> u_prop = propagate(u, prop, Forward);
```

See also: [`ASPropZ`](@ref), [`TiltedASProp`](@ref), [`ParaxialProp`](@ref)
"""
struct ASProp{M, C} <: AbstractSequence{M}
    optical_components::C

    function ASProp(optical_components::C) where {C}
        new{Trainable, C}(optical_components)
    end

    function ASProp(u::ScalarField{U, Nd},
                    ds::NTuple{Nd, Real},
                    z::Real;
                    use_cache::Bool = true,
                    n0::Real = 1,
                    filter = nothing,
                    paraxial::Bool = false,
                    double_precision_kernel::Bool = use_cache) where {U, Nd}
        kernel = ASKernel(u, ds, z; use_cache, n0, filter, paraxial,
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
                    n0::Real = 1,
                    filter = nothing,
                    paraxial::Bool = false,
                    double_precision_kernel::Bool = use_cache)
        ASProp(u, u.ds, z; use_cache, n0, filter, paraxial, double_precision_kernel)
    end
end

Functors.@functor ASProp (optical_components,)

get_sequence(p::ASProp) = p.optical_components

"""
    ASPropZ(u::ScalarField, ds::NTuple, z::Real; n0=1, trainable=false, paraxial=false, filter=nothing, double_precision_kernel=false)
    ASPropZ(u::ScalarField, z::Real; n0=1, trainable=false, paraxial=false, filter=nothing, double_precision_kernel=false)

Angular Spectrum propagation with trainable distance parameter.

Similar to `ASProp` but with a trainable propagation distance `z`. This is useful
for optimization problems where the exact propagation distance is uncertain or
needs to be optimized. Compatible with any Julia automatic differentiation library.

# Arguments
- `u::ScalarField`: Reference field for grid definition
- `ds::NTuple`: Custom spatial sampling intervals (first form only, defaults to `u.ds` in second form)
- `z::Real`: Initial propagation distance
- `n0=1`: Refractive index of medium
- `trainable=false`: Make the distance parameter trainable
- `paraxial=false`: Use paraxial approximation  
- `filter=nothing`: Optional spatial frequency filter
- `double_precision_kernel=false`: Use Float64 for kernel computation

# Returns
Pure component with trainable distance (if `trainable=true`).

# Performance Note
This implementation relies on automatic differentiation libraries (Zygote, Enzyme, etc.)
for gradient computation, which may be slower than the static `ASProp` for non-trainable 
cases. Use `ASProp` when `z` is fixed.

# Examples
```jldoctest
julia> u = ScalarField(ones(ComplexF64, 64, 64), (2.0, 2.0), 1.064);

julia> prop_z = ASPropZ(u, 500.0; trainable=true);

julia> prop_z_custom = ASPropZ(u, (1.5, 1.5), 500.0; trainable=true);

julia> u_prop = propagate(u, prop_z, Forward);

# Distance can be optimized via gradients
julia> OpticalComponents.trainable(prop_z).z isa AbstractArray
true
```

See also: [`ASProp`](@ref), [`trainable`](@ref)
"""
struct ASPropZ{M, T, A, V, H} <: AbstractPureComponent{M}
    n0::T
    z::A
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
        new{M, Tp, typeof(z_arr), V, H}(Tp(n0), z_arr, paraxial, f_vec, filter)
    end

    function ASPropZ(u::ScalarField,
                     z::Real;
                     n0::Real = 1,
                     trainable::Bool = false,
                     paraxial::Bool = false,
                     filter = nothing,
                     double_precision_kernel::Bool = false)
        ASPropZ(u, u.ds, z; n0, trainable, paraxial, filter, double_precision_kernel)
    end
end

Functors.@functor ASPropZ (z,)

trainable(p::ASPropZ{<:Trainable}) = (; z = p.z)

function propagate(u::ScalarField, p::ASPropZ, direction::Type{<:Direction})
    ndims = length(p.f_vec)
    dims = ntuple(i -> i, ndims)
    lambdas = get_lambdas(u)
    if p.is_paraxial
        kernel = @. as_paraxial_kernel(p.f_vec..., lambdas, p.n0, p.z, p.filter)
    else
        z_pos = Val(all(sign.(p.z) .> 0))
        kernel = @. as_kernel(p.f_vec..., lambdas, p.n0, p.z, p.filter, z_pos)
    end
    data = ifft(fft(u.electric, dims) .* conj_direction(kernel, direction), dims)
    set_field_data(u, data)
end
