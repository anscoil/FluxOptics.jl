function as_kernel(fx::T, fy::T, λ::T, n0::Tp, z::Tp,
        filter::H, z_pos::Val{true}) where {T <: Real, Tp <: Real, H}
    fx, fy, λ = Tp(fx), Tp(fy), Tp(λ)/n0
    f² = complex(inv(λ^2))
    v = isnothing(filter) ? Complex{Tp}(1) : Complex{Tp}(filter(fx, fy))
    Complex{T}(cis(Tp(2)*π*z*sqrt(f² - fx^2 - fy^2)) * v)
end

function as_kernel(fx::T, fy::T, λ::T, n0::Tp, z::Tp,
        filter::H, z_pos::Val{false}) where {T <: Real, Tp <: Real, H}
    fx, fy, λ = Tp(fx), Tp(fy), Tp(λ)/n0
    f² = complex(inv(λ^2))
    v = isnothing(filter) ? Complex{Tp}(1) : Complex{Tp}(filter(fx, fy))
    Complex{T}(conj(cis(Tp(2)*π*(-z)*sqrt(f² - fx^2 - fy^2)) * v))
end

function as_kernel(fx::T, λ::T, n0::Tp, z::Tp, filter::H,
        z_pos::Val{true}
) where {T <: Real, Tp <: Real, H}
    fx, λ = Tp(fx), Tp(λ)/n0
    f² = complex(inv(λ)^2)
    v = isnothing(filter) ? Complex{Tp}(1) : Complex{Tp}(filter(fx))
    Complex{T}(cis(Tp(2)*π*z*sqrt(f² - fx^2)) * v)
end

function as_kernel(fx::T, λ::T, n0::Tp, z::Tp, filter::H,
        z_pos::Val{false}
) where {T <: Real, Tp <: Real, H}
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

function as_paraxial_kernel(fx::T, λ::T, n0::Tp, z::Tp, filter::H
) where {T <: Real, Tp <: Real, H}
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
            double_precision_kernel::Bool = use_cache
    ) where {Nd, T, H, U <: AbstractArray{Complex{T}}}
        ns = size(u)[1:Nd]
        cache_size = use_cache ? prod(size(u)[(Nd + 1):end]) : 0
        kernel = FourierKernel(u.data, ns, ds, cache_size)
        Tp = double_precision_kernel ? Float64 : T
        new{Static, typeof(kernel), T, Tp, H}(kernel, paraxial, Tp(n0), Tp(z), filter)
    end
end

Functors.@functor ASKernel ()

get_kernels(p::ASKernel) = (p.kernel,)

build_kernel_key_args(p::ASKernel, u::ScalarField) = (select_lambdas(u),)

function build_kernel_args(p::ASKernel)
    if p.is_paraxial
        (p.n0, p.z, p.filter)
    else
        (p.n0, p.z, p.filter, Val(sign(p.z) > 0))
    end
end

function _propagate_core!(
        apply_kernel_fns::F, u::AbstractArray, p::ASKernel, ::Type{<:Direction}) where {F}
    apply_kernel_fn!, = apply_kernel_fns
    if p.is_paraxial
        apply_kernel_fn!(u, as_paraxial_kernel)
    else
        apply_kernel_fn!(u, as_kernel)
    end
    u
end

function ASProp(u::ScalarField{U, Nd},
        ds::NTuple{Nd, Real},
        z::Real;
        use_cache::Bool = true,
        n0::Real = 1,
        filter = nothing,
        paraxial::Bool = false,
        double_precision_kernel::Bool = use_cache) where {U, Nd}
    kernel = ASKernel(u, ds, z; use_cache, n0, filter, paraxial, double_precision_kernel)
    FourierWrapper(kernel.kernel.p_f, kernel)
end

function ASProp(u::ScalarField, z::Real;
        use_cache::Bool = true,
        n0::Real = 1,
        filter = nothing,
        paraxial::Bool = false,
        double_precision_kernel::Bool = use_cache)
    ASProp(u, u.ds, z; use_cache, n0, filter, paraxial, double_precision_kernel)
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
            double_precision_kernel::Bool = false
    ) where {Nd, T, H, U <: AbstractArray{Complex{T}}}
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

    function ASPropZ(u::ScalarField, z::Real;
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
    data = ifft(fft(u.data, dims) .* conj_direction(kernel, direction), dims)
    set_field_data(u, data)
end
