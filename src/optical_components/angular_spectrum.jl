function compute_as_kernel(fx::T, fy::T, λ::T, z::Tp,
        filter::H) where {T <: AbstractFloat, Tp <: AbstractFloat, H}
    fx, fy = Tp(fx), Tp(fy)
    f² = complex(inv(Tp(λ))^2)
    v = isnothing(filter) ? Tp(1) : Tp(filter(fx, fy))
    cis(Tp(2)*π*z*sqrt(f² - fx^2 - fy^2)) * v
end

function compute_as_kernel(fx::T, λ::T, z::Tp, filter::H
) where {T <: AbstractFloat, Tp <: AbstractFloat, H}
    fx, fy = Tp(fx), Tp(fy)
    f² = complex(inv(Tp(λ)^2))
    v = isnothing(filter) ? Tp(1) : Tp(filter(fx))
    cis(Tp(2)*π*z*sqrt(f² - fx^2)) * v
end

struct ASProp{M, K, T, Tp, H} <: AbstractPropagator{M, K}
    kernel::K
    z::Tp
    filter::H

    function ASProp(u::AbstractArray{Complex{T}, N},
            ds::NTuple{Nd, Real},
            z::Real,
            λ::Real;
            filter::H = nothing,
            double_precision_kernel = true
    ) where {N, Nd, T, H}
        @assert N >= Nd
        ns = size(u)[1:Nd]
        kernel = FourierKernel(u, ns, ds, 1)
        kernel_key = hash(T(λ))
        Tp = double_precision_kernel ? Float64 : T
        fill_kernel_cache(kernel, kernel_key, compute_as_kernel, (T(λ), Tp(z), filter))
        new{Static, typeof(kernel), T, Tp, H}(kernel, Tp(z), filter)
    end

    function ASProp(u::ScalarField{U},
            ds::NTuple{Nd, Real},
            z::Real,
            use_cache::Bool = false;
            filter::H = nothing,
            double_precision_kernel = true
    ) where {N, Nd, T, H, U <: AbstractArray{Complex{T}, N}}
        @assert ndims(u.data) >= Nd
        ns = size(u)[1:Nd]
        cache_size = use_cache ? length(unique(u.lambdas)) : 0
        kernel = FourierKernel(u.data, ns, ds, cache_size)
        Tp = double_precision_kernel ? Float64 : T
        new{Static, typeof(kernel), T, Tp, H}(kernel, Tp(z), filter)
    end
end

function _propagate_core!(apply_kernel_fn::F, u::AbstractArray, p::ASProp) where {F}
    p_f = p.kernel.p_f
    p_f.ft * u
    apply_kernel_fn()
    p_f.ift * u
end

function propagate!(u::AbstractArray, p::ASProp{M, <:FourierKernel{K, T}}, λ::Real,
        direction::Type{<:Direction}) where {M, T, K <: AbstractArray}
    kernel_key = hash(T(λ))
    _propagate_core!(u, p) do
        apply_kernel!(
            u, p.kernel, kernel_key, direction, compute_as_kernel, (T(λ), p.z, p.filter))
    end
    u
end

function propagate!(u::AbstractArray, p::ASProp{M, <:FourierKernel{K}},
        direction::Type{<:Direction}) where {M, K <: AbstractArray}
    kernel_cache = p.kernel.kernel_cache
    (!isnothing(kernel_cache) && length(kernel_cache) == 1) ||
        error("Propagation kernel should hold exactly one wavelength")
    kernel_key = first(keys(kernel_cache))
    _propagate_core!(u, p) do
        apply_kernel!(u, p.kernel, kernel_key, direction)
    end
    u
end

function propagate!(u::ScalarField, p::ASProp{M, <:FourierKernel{Nothing}},
        direction::Type{<:Direction}) where {M}
    _propagate_core!(u.data, p) do
        apply_kernel!(u.data, p.kernel, direction, compute_as_kernel,
            (u.lambdas, p.z, p.filter))
    end
    u
end

function propagate!(u::ScalarField, p::ASProp{M, <:FourierKernel{K}},
        direction::Type{<:Direction}) where {M, K <: AbstractArray}
    kernel_keys = hash.(u.lambdas_collection)
    _propagate_core!(u.data, p) do
        apply_kernel!(u.data, p.kernel, kernel_keys, direction,
            compute_as_kernel, (u.lambdas_collection, p.z, p.filter))
    end
    u
end
