function compute_as_kernel(fx::T, fy::T, λ::T, z::Tp,
        filter::H) where {T <: AbstractFloat, Tp <: AbstractFloat, H}
    fx, fy = Tp(fx), Tp(fy)
    f² = complex(inv(Tp(λ))^2)
    v = isnothing(filter) ? Tp(1) : Tp(filter(fx, fy))
    Complex{T}(cis(Tp(2)*π*z*sqrt(f² - fx^2 - fy^2)) * v)
end

function compute_as_kernel(fx::T, λ::T, z::Tp, filter::H
) where {T <: AbstractFloat, Tp <: AbstractFloat, H}
    fx, fy = Tp(fx), Tp(fy)
    f² = complex(inv(Tp(λ)^2))
    v = isnothing(filter) ? Tp(1) : Tp(filter(fx))
    Complex{T}(cis(Tp(2)*π*z*sqrt(f² - fx^2)) * v)
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

function get_kernels(p::ASProp)
    (p.kernel,)
end

function build_kernel_key_args(p::ASProp{M, K, T}, λ::Real) where {M, K, T}
    hash(T(λ)), (T(λ), p.z, p.filter)
end

function build_kernel_args(p::ASProp, u::ScalarField)
    #        p::ASProp{M, <:FourierKernel{Nothing}}, u::ScalarField) where {M}
    (u.lambdas, p.z, p.filter)
end

function build_kernel_key_args(p::ASProp, u::ScalarField)
    # p::ASProp{M, <:FourierKernel{K}},
    # u::ScalarField) where {M, K <: AbstractArray}
    hash.(u.lambdas_collection), (u.lambdas_collection, p.z, p.filter)
end

function _propagate_core!(
        apply_kernel_fns::F, u::AbstractArray, p::ASProp, ::Type{<:Direction}) where {F}
    apply_kernel_fn!, = apply_kernel_fns
    p_f = p.kernel.p_f
    p_f.ft * u
    apply_kernel_fn!(u, compute_as_kernel)
    p_f.ift * u
end
