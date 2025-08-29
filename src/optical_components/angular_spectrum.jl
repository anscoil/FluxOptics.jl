function as_kernel(fx::T, fy::T, λ::T, z::Tp,
        filter::H) where {T <: Real, Tp <: Real, H}
    fx, fy = Tp(fx), Tp(fy)
    f² = complex(inv(Tp(λ))^2)
    v = isnothing(filter) ? Tp(1) : Tp(filter(fx, fy))
    Complex{T}(cis(Tp(2)*π*z*sqrt(f² - fx^2 - fy^2)) * v)
end

function as_kernel(fx::T, λ::T, z::Tp, filter::H
) where {T <: Real, Tp <: Real, H}
    fx, fy = Tp(fx), Tp(fy)
    f² = complex(inv(Tp(λ)^2))
    v = isnothing(filter) ? Tp(1) : Tp(filter(fx))
    Complex{T}(cis(Tp(2)*π*z*sqrt(f² - fx^2)) * v)
end

function as_paraxial_kernel(fx::T, fy::T, λ::T, z::Tp,
        filter::H) where {T <: Real, Tp <: Real, H}
    fx, fy, λ = Tp(fx), Tp(fy), Tp(λ)
    v = isnothing(filter) ? Tp(1) : Tp(filter(fx, fy))
    Complex{T}(cis(-π*λ*z*(fx^2 + fy^2)) * v)
end

function as_paraxial_kernel(fx::T, λ::T, z::Tp, filter::H
) where {T <: Real, Tp <: Real, H}
    fx, λ = Tp(fx), Tp(λ)
    v = isnothing(filter) ? Tp(1) : Tp(filter(fx))
    Complex{T}(cis(-π*λ*z*fx^2) * v)
end

struct ASProp{M, K, T, Tp, H} <: AbstractPropagator{M, K}
    kernel::K
    is_paraxial::Bool
    z::Tp
    filter::H

    function ASProp(u::AbstractArray{Complex{T}, N},
            ds::NTuple{Nd, Real},
            z::Real,
            λ::Real;
            filter::H = nothing,
            paraxial::Bool = false,
            double_precision_kernel::Bool = true
    ) where {N, Nd, T, H}
        @assert N >= Nd
        ns = size(u)[1:Nd]
        kernel = FourierKernel(u, ns, ds, 1)
        kernel_key = hash(T(λ))
        Tp = double_precision_kernel ? Float64 : T
        compute_kernel = paraxial ? as_paraxial_kernel : as_kernel
        fill_kernel_cache(kernel, kernel_key, compute_kernel, (T(λ), Tp(z), filter))
        new{Static, typeof(kernel), T, Tp, H}(kernel, paraxial, Tp(z), filter)
    end

    function ASProp(u::ScalarField{U, Nd},
            z::Real,
            use_cache::Bool = false;
            filter::H = nothing,
            paraxial::Bool = false,
            double_precision_kernel::Bool = true
    ) where {Nd, T, H, U <: AbstractArray{Complex{T}}}
        ns = size(u)[1:Nd]
        cache_size = use_cache ? length(unique(u.lambdas)) : 0
        kernel = FourierKernel(u.data, ns, u.ds, cache_size)
        Tp = double_precision_kernel ? Float64 : T
        new{Static, typeof(kernel), T, Tp, H}(kernel, paraxial, Tp(z), filter)
    end
end

Functors.@functor ASProp ()

function get_kernels(p::ASProp)
    (p.kernel,)
end

function build_kernel_key_args(p::ASProp{M, K, T}, λ::Real) where {M, K, T}
    hash(T(λ)), (T(λ), p.z, p.filter)
end

function build_kernel_args(p::ASProp, u::ScalarField)
    (u.lambdas, p.z, p.filter)
end

function build_kernel_key_args(p::ASProp, u::ScalarField)
    hash.(u.lambdas_collection), (u.lambdas_collection, p.z, p.filter)
end

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

struct ASPropZ{M, A, V, H} <: AbstractPureComponent{M}
    z::A
    is_paraxial::Bool
    f_vec::V
    filter::H

    # After functor destructuring, ASPropZ is always re-instantiated
    # as `Static`.  Even if the original instance was trainable, the
    # restructured version does not require to expose its trainable
    # parameters, since they are no longer optimized.
    function ASPropZ(z::A, is_paraxial::Bool, f_vec::V, filter::H) where {A, V, H}
        new{Static, A, V, H}(z, is_paraxial, f_vec, filter)
    end

    function ASPropZ(u::ScalarField{U, Nd},
            z::Real;
            trainable::Bool = false,
            paraxial::Bool = false,
            filter::H = nothing,
            double_precision_kernel::Bool = true
    ) where {Nd, T, H, U <: AbstractArray{Complex{T}}}
        ns = size(u)[1:Nd]
        F = adapt_dim(U, 1, real)
        fs = [fftfreq(nx, 1/dx) |> F for (nx, dx) in zip(ns, u.ds)]
        f_vec = Nd == 2 ? (; x = fs[1], y = fs[2]') : (; x = fs[1])
        V = typeof(f_vec)
        M = trainable ? Trainable{GradNoAlloc} : Static
        Tp = double_precision_kernel ? Float64 : T
        z_arr = Tp.([z] |> F)
        new{M, typeof(z_arr), V, H}(z_arr, paraxial, f_vec, filter)
    end
end

Functors.@functor ASPropZ (z,)

trainable(p::ASPropZ{<:Trainable}) = (; z = p.z)

function propagate(u::ScalarField, p::ASPropZ, direction::Type{<:Direction})
    ndims = length(p.f_vec)
    dims = ntuple(i -> i, ndims)
    if p.is_paraxial
        kernel = @. as_paraxial_kernel(p.f_vec..., u.lambdas, p.z, p.filter)
    else
        kernel = @. as_kernel(p.f_vec..., u.lambdas, p.z, p.filter)
    end
    data = ifft(fft(u.data, dims) .* kernel_direction(kernel, direction), dims)
    ScalarField(data, u.ds, u.lambdas, u.lambdas_collection)
end
