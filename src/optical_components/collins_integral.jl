function collins_a_chirp(
        x::T, y::T, λ::T, αx::Tp, αy::Tp, a::Tp, b::Tp, d::Tp
) where {T <: AbstractFloat, Tp <: AbstractFloat}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    Complex{T}(cis(π*(x^2*αx*(a*αx-1) + y^2*αy*(a*αy-1))/(b*λ))/λ)
end

function collins_d_chirp(
        x::T, y::T, λ::T, αx::Tp, αy::Tp, a::Tp, b::Tp, d::Tp
) where {T <: AbstractFloat, Tp <: AbstractFloat}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    Complex{T}(cis(π*(x^2*(d-αx) + y^2*(d-αy))/(b*λ)))
end

function collins_convolution_kernel(
        x::T, y::T, λ::T, αx::Tp, αy::Tp, a::Tp, b::Tp, d::Tp
) where {T <: AbstractFloat, Tp <: AbstractFloat}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    Complex{T}(cis(π*(x^2*αx + y^2*αy)/(b*λ)))
end

struct CollinsKernel{K, V, P, U} <: AbstractKernel{K, V, 2}
    a_chirp::ChirpKernel{K, V}
    d_chirp::ChirpKernel{K, V}
    convolution_kernel::ConvolutionKernel{K, V, P, U}
end

struct CollinsProp{M, K, T, Tp, Nd} <: AbstractPropagator{M, K}
    kernel::K
    αs::NTuple{Nd, Tp}
    abd::Tuple{Tp, Tp, Tp}
    nrm_fwd::Complex{Tp}
    nrm_bwd::Complex{Tp}

    function CollinsProp(u::AbstractArray{Complex{T}, N},
            ds::NTuple{Nd, Real},
            ds′::NTuple{Nd, Real},
            abd::Tuple{<:Real, <:Real, <:Real},
            λ::Real;
            double_precision_kernel = true
    ) where {N, Nd, T}
        @assert N >= Nd
        ns = size(u)[1:Nd]
        a_chirp = ChirpKernel(u, ns, ds, 1)
        d_chirp = ChirpKernel(u, ns, ds, 1)
        conv_kernel = ConvolutionKernel(u, ns, ds, 1)
        kernel = CollinsKernel(a_chirp, d_chirp, conv_kernel)
        kernel_key = hash(T(λ))
        Tp = double_precision_kernel ? Float64 : T
        αs = Tuple([Tp(dx′/dx) for (dx, dx′) in zip(ds, ds′)])
        _, b = abd
        nrm_fwd = Complex{Tp}(prod(ds .* sqrt(-im/b)))
        nrm_bwd = Complex{Tp}(prod(ds′ .* sqrt(im/b)))
        abd = Tp.(abd)
        kernel_args = (T(λ), αs..., abd...)
        fill_kernel_cache(a_chirp, kernel_key, collins_a_chirp, kernel_args)
        fill_kernel_cache(d_chirp, kernel_key, collins_d_chirp, kernel_args)
        fill_kernel_cache(conv_kernel, kernel_key, collins_convolution_kernel, kernel_args)
        new{Static, typeof(kernel), T, Tp, Nd}(kernel, αs, abd, nrm_fwd, nrm_bwd)
    end

    function CollinsProp(u::ScalarField{U},
            ds::NTuple{Nd, Real},
            ds′::NTuple{Nd, Real},
            abd::Tuple{<:Real, <:Real, <:Real},
            use_cache::Bool = false;
            double_precision_kernel = true
    ) where {N, Nd, T, U <: AbstractArray{Complex{T}, N}}
        @assert ndims(u.data) >= Nd
        ns = size(u)[1:Nd]
        cache_size = use_cache ? length(unique(u.lambdas)) : 0
        a_chirp = ChirpKernel(u, ns, ds, cache_size)
        d_chirp = ChirpKernel(u, ns, ds, cache_size)
        conv_kernel = ConvolutionKernel(u, ns, ds, cache_size)
        kernel = CollinsKernel(a_chirp, d_chirp, conv_kernel)
        Tp = double_precision_kernel ? Float64 : T
        αs = Tuple([Tp(dx′/dx) for (dx, dx′) in zip(ds, ds′)])
        _, b = abd
        nrm_fwd = Complex{Tp}(prod(ds .* sqrt(-im/b)))
        nrm_bwd = Complex{Tp}(prod(ds′ .* sqrt(im/b)))
        new{Static, typeof(kernel), T, Tp, Nd}(kernel, αs, Tp.(abd), nrm_fwd, nrm_bwd)
    end
end

function get_kernels(p::CollinsProp)
    (p.kernel.a_chirp, p.kernel.d_chirp, p.kernel.convolution_kernel)
end

function build_kernel_key_args(p::CollinsProp{M, K, T}, λ::Real) where {M, K, T}
    hash(T(λ)), (T(λ), p.αs..., p.abd...)
end

function build_kernel_args(p::CollinsProp, u::ScalarField)
    (u.lambdas, p.αs..., p.abd...)
end

function build_kernel_key_args(p::CollinsProp, u::ScalarField)
    hash.(u.lambdas_collection), (u.lambdas_collection, p.αs..., p.abd...)
end

function apply_collins_first_chirp!(u_tmp, apply_a_chirp!, apply_d_chirp!, ::Type{Forward})
    apply_d_chirp!(u_tmp, collins_d_chirp)
end

function apply_collins_first_chirp!(u_tmp, apply_a_chirp!, apply_d_chirp!, ::Type{Backward})
    apply_a_chirp!(u_tmp, collins_a_chirp)
end

function apply_collins_last_chirp!(u_tmp, apply_a_chirp!, apply_d_chirp!, ::Type{Forward})
    apply_a_chirp!(u_tmp, collins_a_chirp)
end

function apply_collins_last_chirp!(u_tmp, apply_a_chirp!, apply_d_chirp!, ::Type{Backward})
    apply_d_chirp!(u_tmp, collins_d_chirp)
end

function normalize!(u::AbstractArray, p::CollinsProp, ::Type{Forward})
    u .*= p.nrm_fwd
end

function normalize!(u::AbstractArray, p::CollinsProp, ::Type{Backward})
    u .*= p.nrm_bwd
end

function _propagate_core!(apply_kernel_fns::F, u::AbstractArray,
        p::CollinsProp, direction::Type{<:Direction}) where {F}
    apply_a_chirp!, apply_d_chirp!, apply_kernel_fn! = apply_kernel_fns
    p_f = p.kernel.convolution_kernel.p_f
    u_tmp = p.kernel.convolution_kernel.u_plan
    u_tmp .= 0
    u_view = view(u_tmp, axes(u)...)
    copyto!(u_view, u)
    apply_collins_first_chirp!(u_tmp, apply_a_chirp!, apply_d_chirp!, direction)
    p_f.ft * u_tmp
    apply_kernel_fn!(u_tmp, collins_convolution_kernel)
    p_f.ift * u_tmp
    apply_collins_last_chirp!(u_tmp, apply_a_chirp!, apply_d_chirp!, direction)
    copyto!(u, u_view)
    normalize!(u, p, direction)
    u
end
