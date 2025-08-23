function fourier_lens_convolution(
        x::T, y::T, λ::T, θx::Tp, θy::Tp, fl::Tp,
        nrm_f::Tp) where {T <: AbstractFloat, Tp <: AbstractFloat}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    Complex{T}(cis((x^2*θx + y^2*θy)/(λ*fl))*nrm_f/λ)
end

function fourier_lens_convolution(
        x::T, λ::T, θx::Tp, fl::Tp,
        nrm_f::Tp) where {T <: AbstractFloat, Tp <: AbstractFloat}
    x, λ = Tp(x), Tp(λ)
    Complex{T}(cis(x^2*θx/(λ*fl))*nrm_f/sqrt(λ))
end

function fourier_lens_chirp(
        x::T, y::T, λ::T, θx::Tp, θy::Tp, fl::Tp,
        ::Tp) where {T <: AbstractFloat, Tp <: AbstractFloat}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    Complex{T}(cis(-(x^2*θx + y^2*θy)/(λ*fl)))
end

function fourier_lens_chirp(
        x::T, λ::T, θx::Tp, fl::Tp,
        ::Tp) where {T <: AbstractFloat, Tp <: AbstractFloat}
    x, λ = Tp(x), Tp(λ)
    Complex{T}(cis(-x^2*θx/(λ*fl)))
end

struct FourierLensKernel{K, V, P, U} <: AbstractKernel{K, V, 2}
    convolution_kernel::ConvolutionKernel{K, V, P, U}
    chirp_kernel::ChirpKernel{K, V}
end

struct FourierLens{M, K, T, Tp, Nd} <: AbstractPropagator{M, K}
    kernel::K
    θs::NTuple{Nd, Tp}
    fl::Tp
    nrm_f::Tp

    # Warning: aliasing expected if nx*dx*dx′/(λ*fl) > 2 || ny*dy*dy′/(λ*fl) > 2
    # but this should not be a relevant use case.
    function FourierLens(u::AbstractArray{Complex{T}, N},
            ds::NTuple{Nd, Real},
            ds′::NTuple{Nd, Real},
            fl::Real,
            λ::Real;
            double_precision_kernel = true
    ) where {N, Nd, T}
        @assert N >= Nd
        ns = size(u)[1:Nd]
        conv_kernel = ConvolutionKernel(u, ns, ds, 1)
        chirp_kernel = ChirpKernel(u, ns, ds, 1)
        kernel = FourierLensKernel(conv_kernel, chirp_kernel)
        kernel_key = hash(T(λ))
        Tp = double_precision_kernel ? Float64 : T
        θs = Tuple([Tp(π*dx′/dx) for (dx, dx′) in zip(ds, ds′)])
        nrm_f = Tp(prod(ds)/fl)
        kernel_args = (T(λ), θs..., Tp(fl), nrm_f)
        fill_kernel_cache(conv_kernel, kernel_key, fourier_lens_convolution, kernel_args)
        fill_kernel_cache(chirp_kernel, kernel_key, fourier_lens_chirp, kernel_args)
        new{Static, typeof(kernel), T, Tp, Nd}(kernel, θs, Tp(fl), nrm_f)
    end

    function FourierLens(u::ScalarField{U},
            ds::NTuple{Nd, Real},
            ds′::NTuple{Nd, Real},
            fl::Real,
            use_cache::Bool = false;
            double_precision_kernel = true
    ) where {N, Nd, T, U <: AbstractArray{Complex{T}, N}}
        @assert ndims(u.data) >= Nd
        ns = size(u)[1:Nd]
        cache_size = use_cache ? length(unique(u.lambdas)) : 0
        conv_kernel = ConvolutionKernel(u.data, ns, ds, cache_size)
        chirp_kernel = ChirpKernel(u.data, ns, ds, cache_size)
        kernel = FourierLensKernel(conv_kernel, chirp_kernel)
        Tp = double_precision_kernel ? Float64 : T
        θs = Tuple([Tp(π*dx′/dx) for (dx, dx′) in zip(ds, ds′)])
        nrm_f = Tp(prod(ds)/fl)
        new{Static, typeof(kernel), T, Tp, Nd}(kernel, θs, Tp(fl), nrm_f)
    end
end

function get_kernels(p::FourierLens)
    (p.kernel.convolution_kernel, p.kernel.chirp_kernel)
end

function build_kernel_key_args(p::FourierLens{M, K, T}, λ::Real) where {M, K, T}
    hash(T(λ)), (T(λ), p.θs..., p.fl, p.nrm_f)
end

function build_kernel_args(p::FourierLens, u::ScalarField)
    (u.lambdas, p.θs..., p.fl, p.nrm_f)
end

function build_kernel_key_args(p::FourierLens, u::ScalarField)
    hash.(u.lambdas_collection), (u.lambdas_collection, p.θs..., p.fl, p.nrm_f)
end

function _propagate_core!(apply_kernel_fns::F, u::AbstractArray,
        p::FourierLens, ::Type{<:Direction}) where {F}
    apply_kernel_fn!, apply_chirp! = apply_kernel_fns
    p_f = p.kernel.convolution_kernel.p_f
    u_tmp = p.kernel.convolution_kernel.u_plan
    u_tmp .= 0
    u_view = view(u_tmp, axes(u)...)
    copyto!(u_view, u)
    apply_chirp!(u_tmp, fourier_lens_chirp)
    p_f.ft * u_tmp
    apply_kernel_fn!(u_tmp, fourier_lens_convolution)
    p_f.ift * u_tmp
    apply_chirp!(u_tmp, fourier_lens_chirp)
    copyto!(u, u_view)
    u
end
