function rs_kernel(x::T, y::T, λ::T, z::Tp, nrm_f::Tp
) where {T <: Real, Tp <: Real}
    x, y = Tp(x), Tp(y)
    k = Tp(2π/λ)
    r = sqrt(x^2 + y^2 + z^2)
    Complex{T}(nrm_f*(cis(k*r)/r)*(z/r)*(1/r-im*k))
end

struct RSProp{M, K, T, Tp} <: AbstractPropagator{M, K, T}
    kernel::K
    z::Tp
    nrm_f::Tp

    function RSProp(u::AbstractArray{Complex{T}, N},
            ds::NTuple{Nd, Real},
            z::Real,
            λ::Real;
            double_precision_kernel::Bool = true
    ) where {N, Nd, T}
        @assert N >= Nd
        @assert z >= 0
        ns = size(u)[1:Nd]
        kernel = ConvolutionKernel(u, ns, ds, 1)
        kernel_key = hash(T(λ))
        Tp = double_precision_kernel ? Float64 : T
        nrm_f = Tp(prod(ds)/2π)
        fill_kernel_cache(kernel, kernel_key, rs_kernel, (T(λ), Tp(z), nrm_f))
        new{Static, typeof(kernel), T, Tp}(kernel, Tp(z), nrm_f)
    end

    function RSProp(u::ScalarField{U, Nd},
            ds::NTuple{Nd, Real},
            z::Real,
            use_cache::Bool = false;
            double_precision_kernel::Bool = true
    ) where {Nd, T, U <: AbstractArray{Complex{T}}}
        @assert z >= 0
        ns = size(u)[1:Nd]
        cache_size = use_cache ? length(unique(u.lambdas)) : 0
        kernel = ConvolutionKernel(u.data, ns, ds, cache_size)
        Tp = double_precision_kernel ? Float64 : T
        nrm_f = Tp(prod(ds)/2π)
        new{Static, typeof(kernel), T, Tp}(kernel, Tp(z), nrm_f)
    end

    function RSProp(u::ScalarField, z::Real, use_cache::Bool = false;
            double_precision_kernel::Bool = true)
        RSProp(u, u.ds, z, use_cache; double_precision_kernel)
    end
end

Functors.@functor RSProp ()

get_kernels(p::RSProp) = (p.kernel,)

build_kernel_keys(p::RSProp{M, K, T}, λ::Real) where {M, K, T} = hash(T(λ))

build_kernel_keys(p::RSProp, lambdas::AbstractArray) = (1, hash.(lambdas))

build_kernel_args(p::RSProp) = (p.z, p.nrm_f)

build_kernel_args_dict(p::RSProp) = build_kernel_args(p::RSProp)

function _propagate_core!(
        apply_kernel_fns::F, u::AbstractArray, p::RSProp, ::Type{<:Direction}) where {F}
    apply_kernel_fn!, = apply_kernel_fns
    p_f = p.kernel.p_f
    u_tmp = p.kernel.u_plan
    u_tmp .= 0
    u_view = view(u_tmp, axes(u)...)
    copyto!(u_view, u)
    p_f.ft * u_tmp
    apply_kernel_fn!(u_tmp, rs_kernel)
    p_f.ift * u_tmp
    copyto!(u, u_view)
    u
end
