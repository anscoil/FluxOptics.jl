function as_kernel(fx::T, fy::T, λ::T, n0::Tp, z::Tp,
        filter::H) where {T <: Real, Tp <: Real, H}
    fx, fy, λ = Tp(fx), Tp(fy), Tp(λ)/n0
    f² = complex(inv(λ^2))
    v = isnothing(filter) ? Tp(1) : Tp(filter(fx, fy))
    Complex{T}(cis(Tp(2)*π*z*sqrt(f² - fx^2 - fy^2)) * v)
end

function as_kernel(fx::T, λ::T, n0::Tp, z::Tp, filter::H
) where {T <: Real, Tp <: Real, H}
    fx, λ = Tp(fx), Tp(λ)/n0
    f² = complex(inv(λ)^2)
    v = isnothing(filter) ? Tp(1) : Tp(filter(fx))
    Complex{T}(cis(Tp(2)*π*z*sqrt(f² - fx^2)) * v)
end

function as_paraxial_kernel(fx::T, fy::T, λ::T, n0::T, z::Tp,
        filter::H) where {T <: Real, Tp <: Real, H}
    fx, fy, λ = Tp(fx), Tp(fy), Tp(λ/n0)
    v = isnothing(filter) ? Tp(1) : Tp(filter(fx, fy))
    Complex{T}(cis(-π*λ*z*(fx^2 + fy^2)) * v)
end

function as_paraxial_kernel(fx::T, λ::T, n0::Tp, z::Tp, filter::H
) where {T <: Real, Tp <: Real, H}
    fx, λ = Tp(fx), Tp(λ/n0)
    v = isnothing(filter) ? Tp(1) : Tp(filter(fx))
    Complex{T}(cis(-π*λ*z*fx^2) * v)
end

struct ASProp{M, K, T, Tp, H} <: AbstractPropagator{M, K, T}
    kernel::K
    is_paraxial::Bool
    n0::Tp
    z::Tp
    filter::H

    function ASProp(u::AbstractArray{Complex{T}, N},
            ds::NTuple{Nd, Real},
            z::Real,
            λ::Real;
            n0::Real = 1,
            filter::H = nothing,
            paraxial::Bool = false,
            double_precision_kernel::Bool = true
    ) where {N, Nd, T, H}
        @assert N >= Nd
        @assert z >= 0
        ns = size(u)[1:Nd]
        kernel = FourierKernel(u, ns, ds, 1)
        kernel_key = hash(T(λ))
        Tp = double_precision_kernel ? Float64 : T
        compute_kernel = paraxial ? as_paraxial_kernel : as_kernel
        fill_kernel_cache(kernel, kernel_key, compute_kernel, (T(λ), Tp(n0), Tp(z), filter))
        new{Static, typeof(kernel), T, Tp, H}(kernel, paraxial, Tp(n0), Tp(z), filter)
    end

    function ASProp(u::ScalarField{U, Nd},
            ds::NTuple{Nd, Real},
            z::Real,
            use_cache::Bool = false;
            n0::Real = 1,
            filter::H = nothing,
            paraxial::Bool = false,
            double_precision_kernel::Bool = true
    ) where {Nd, T, H, U <: AbstractArray{Complex{T}}}
        @assert z >= 0
        ns = size(u)[1:Nd]
        cache_size = use_cache ? length(unique(u.lambdas)) : 0
        kernel = FourierKernel(u.data, ns, ds, cache_size)
        Tp = double_precision_kernel ? Float64 : T
        new{Static, typeof(kernel), T, Tp, H}(kernel, paraxial, Tp(n0), Tp(z), filter)
    end

    function ASProp(u::ScalarField, z::Real, use_cache::Bool = false;
            n0::Real = 1,
            filter = nothing,
            paraxial::Bool = false,
            double_precision_kernel::Bool = true)
        ASProp(u, u.ds, z, use_cache; n0, filter, paraxial, double_precision_kernel)
    end
end

Functors.@functor ASProp ()

get_kernels(p::ASProp) = (p.kernel,)

build_kernel_keys(p::ASProp{M, K, T}, λ::Real) where {M, K, T} = hash(T(λ))

build_kernel_keys(p::ASProp, lambdas::AbstractArray) = (1, hash.(lambdas))

build_kernel_args(p::ASProp) = (p.n0, p.z, p.filter)

build_kernel_args_dict(p::ASProp) = build_kernel_args(p::ASProp)

function _propagate_core!(
        apply_kernel_fns::F, u::AbstractArray, p::ASProp, ::Type{<:Direction}) where {F}
    apply_kernel_fn!, = apply_kernel_fns
    p_f = p.kernel.p_f
    p_f.ft * u
    if p.is_paraxial
        apply_kernel_fn!(u, as_paraxial_kernel)
    else
        apply_kernel_fn!(u, as_kernel)
    end
    p_f.ift * u
end

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
            double_precision_kernel::Bool = true
    ) where {Nd, T, H, U <: AbstractArray{Complex{T}}}
        @assert z >= 0
        ns = size(u)[1:Nd]
        F = adapt_dim(U, 1, real)
        fs = [fftfreq(nx, 1/dx) |> F for (nx, dx) in zip(ns, ds)]
        f_vec = Nd == 2 ? (; x = fs[1], y = fs[2]') : (; x = fs[1])
        V = typeof(f_vec)
        M = trainable ? Trainable : Static
        Tp = double_precision_kernel ? Float64 : T
        z_arr = Tp.([z] |> F)
        new{M, Tp, typeof(z_arr), V, H}(Tp(n0), z_arr, paraxial, f_vec, filter)
    end

    function ASPropZ(u::ScalarField, z::Real;
            n0::Real = 1,
            trainable::Bool = false,
            paraxial::Bool = false,
            filter = nothing,
            double_precision_kernel::Bool = true)
        ASPropZ(u, u.ds, z; n0, trainable, paraxial, filter, double_precision_kernel)
    end
end

Functors.@functor ASPropZ (z,)

trainable(p::ASPropZ{<:Trainable}) = (; z = p.z)

function propagate(u::ScalarField, p::ASPropZ, direction::Type{<:Direction})
    ndims = length(p.f_vec)
    dims = ntuple(i -> i, ndims)
    if p.is_paraxial
        kernel = @. as_paraxial_kernel(p.f_vec..., u.lambdas/p.n0, p.z, p.filter)
    else
        kernel = @. as_kernel(p.f_vec..., u.lambdas/p.n0, p.z, p.filter)
    end
    data = ifft(fft(u.data, dims) .* conj_direction(kernel, direction), dims)
    ScalarField(data, u.ds, u.lambdas, u.lambdas_collection)
end
