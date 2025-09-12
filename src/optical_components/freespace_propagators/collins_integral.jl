# Stuart A. Collins, "Lens-System Diffraction Integral Written in
# Terms of Matrix Optics*," J. Opt. Soc. Am. 60, 1168-1177 (1970)
# https://doi.org/10.1364/JOSA.60.001168

function collins_a_chirp(
        x::T, y::T, λ::T, αx::Tp, αy::Tp, a::Tp, b::Tp,
        d::Tp
) where {T <: Real, Tp <: Real}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    Complex{T}(cis(π*(x^2*(a-αx) + y^2*(a-αy))/(b*λ))/λ)
end

function collins_d_chirp(
        x::T, y::T, λ::T, αx::Tp, αy::Tp, a::Tp, b::Tp,
        d::Tp
) where {T <: Real, Tp <: Real}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    Complex{T}(cis(π*(x^2*αx*(d*αx-1) + y^2*αy*(d*αy-1))/(b*λ)))
end

function collins_convolution_kernel(
        x::T, y::T, λ::T, αx::Tp, αy::Tp, a::Tp, b::Tp,
        d::Tp
) where {T <: Real, Tp <: Real}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    Complex{T}(cis(π*(x^2*αx + y^2*αy)/(b*λ)))
end

struct CollinsKernel{K, V, P, U} <: AbstractKernel{K, V}
    a_chirp::ChirpKernel{K, V}
    d_chirp::ChirpKernel{K, V}
    convolution_kernel::ConvolutionKernel{K, V, P, U}
end

struct CollinsProp{M, K, T, Tp, Nd} <: AbstractPropagator{M, K, T}
    kernel::K
    αs::NTuple{Nd, Tp}
    ds::NTuple{Nd, Tp}
    ds′::NTuple{Nd, Tp}
    abd::Tuple{Tp, Tp, Tp}
    nrm_fwd::Complex{Tp}
    nrm_bwd::Complex{Tp}

    function CollinsProp(u::ScalarField{U, Nd},
            ds::NTuple{Nd, Real},
            ds′::NTuple{Nd, Real},
            abd::Tuple{<:Real, <:Real, <:Real};
            use_cache::Bool = true,
            double_precision_kernel::Bool = use_cache
    ) where {N, Nd, T, U <: AbstractArray{Complex{T}, N}}
        @assert N >= Nd
        ns = size(u)[1:Nd]
        cache_size = use_cache ? prod(size(u)[(Nd + 1):end]) : 0
        a_chirp = ChirpKernel(u.data, ns, ds, cache_size)
        d_chirp = ChirpKernel(u.data, ns, ds, cache_size)
        conv_kernel = ConvolutionKernel(u.data, ns, ds, cache_size)
        kernel = CollinsKernel(a_chirp, d_chirp, conv_kernel)
        Tp = double_precision_kernel ? Float64 : T
        αs = Tuple([Tp(dx′/dx) for (dx, dx′) in zip(ds, ds′)])
        _, b = abd
        nrm_fwd = Complex{Tp}(prod(ds ./ sqrt(im*b)))
        nrm_bwd = Complex{Tp}(prod(ds′ ./ sqrt(-im*b)))
        new{Static, typeof(kernel), T, Tp, Nd}(
            kernel, αs, ds, ds′, Tp.(abd), nrm_fwd, nrm_bwd)
    end

    function CollinsProp(u::ScalarField{U, Nd},
            ds′::NTuple{Nd, Real},
            abd::Tuple{<:Real, <:Real, <:Real};
            use_cache::Bool = true,
            double_precision_kernel::Bool = use_cache
    ) where {Nd, U <: AbstractArray{<:Complex}}
        CollinsProp(u, u.ds, ds′, abd; use_cache, double_precision_kernel)
    end
end

Functors.@functor CollinsProp ()

function get_kernels(p::CollinsProp)
    (p.kernel.a_chirp, p.kernel.d_chirp, p.kernel.convolution_kernel)
end

build_kernel_key_args(p::CollinsProp, u::ScalarField) = (select_lambdas(u),)

build_kernel_args(p::CollinsProp) = (p.αs..., p.abd...)

function apply_collins_first_chirp!(u_tmp, apply_a_chirp!, apply_d_chirp!, ::Type{Forward})
    apply_a_chirp!(u_tmp, collins_a_chirp)
end

function apply_collins_first_chirp!(u_tmp, apply_a_chirp!, apply_d_chirp!, ::Type{Backward})
    apply_d_chirp!(u_tmp, collins_d_chirp)
end

function apply_collins_last_chirp!(u_tmp, apply_a_chirp!, apply_d_chirp!, ::Type{Forward})
    apply_d_chirp!(u_tmp, collins_d_chirp)
end

function apply_collins_last_chirp!(u_tmp, apply_a_chirp!, apply_d_chirp!, ::Type{Backward})
    apply_a_chirp!(u_tmp, collins_a_chirp)
end

function normalize_collins!(u::AbstractArray, p::CollinsProp, ::Type{Forward})
    u .*= p.nrm_fwd
end

function normalize_collins!(u::AbstractArray, p::CollinsProp, ::Type{Backward})
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
    normalize_collins!(u, p, direction)
    u
end

function set_ds_out(p::CollinsProp, u::ScalarField, ::Type{Forward})
    set_field_ds(u, p.ds′)
end

function set_ds_out(p::CollinsProp, u::ScalarField, ::Type{Backward})
    set_field_ds(u, p.ds)
end
