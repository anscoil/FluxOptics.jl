function rs_kernel(x::T, y::T, λ::T, z::Tp, nrm_f::Tp, z_pos::Val{true}
) where {T <: Real, Tp <: Real}
    x, y = Tp(x), Tp(y)
    k = Tp(2π/λ)
    r = sqrt(x^2 + y^2 + z^2)
    Complex{T}(nrm_f*(cis(k*r)/r)*(z/r)*(1/r-im*k))
end

function rs_kernel(x::T, y::T, λ::T, z::Tp, nrm_f::Tp, z_pos::Val{false}
) where {T <: Real, Tp <: Real}
    x, y = Tp(x), Tp(y)
    k = Tp(2π/λ)
    r = sqrt(x^2 + y^2 + z^2)
    Complex{T}(conj(nrm_f*(cis(k*r)/r)*(z/r)*(1/r-im*k)))
end

struct RSProp{M, K, T, Tp} <: AbstractPropagator{M, K, T}
    kernel::K
    z::Tp
    nrm_f::Tp

    function RSProp(u::ScalarField{U, Nd},
            ds::NTuple{Nd, Real},
            z::Real;
            use_cache::Bool = true,
            double_precision_kernel::Bool = true
    ) where {Nd, T, U <: AbstractArray{Complex{T}}}
        ns = size(u)[1:Nd]
        cache_size = use_cache ? prod(size(u)[(Nd + 1):end]) : 0
        kernel = ConvolutionKernel(u.data, ns, ds, cache_size)
        Tp = double_precision_kernel ? Float64 : T
        nrm_f = Tp(prod(ds)/2π)
        new{Static, typeof(kernel), T, Tp}(kernel, Tp(z), nrm_f)
    end

    function RSProp(u::ScalarField, z::Real; use_cache::Bool = true,
            double_precision_kernel::Bool = true)
        RSProp(u, u.ds, z; use_cache, double_precision_kernel)
    end
end

Functors.@functor RSProp ()

get_kernels(p::RSProp) = (p.kernel,)

build_kernel_key_args(p::RSProp, u::ScalarField) = (select_lambdas(u),)

build_kernel_args(p::RSProp) = (p.z, p.nrm_f, Val(sign(p.z) > 0))

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
