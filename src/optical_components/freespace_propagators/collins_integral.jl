# Stuart A. Collins, "Lens-System Diffraction Integral Written in
# Terms of Matrix Optics*," J. Opt. Soc. Am. 60, 1168-1177 (1970)
# https://doi.org/10.1364/JOSA.60.001168

function collins_a_chirp(x::T, y::T, λ::T, αx::Tp, αy::Tp, a::Tp, b::Tp,
                         d::Tp) where {T <: Real, Tp <: Real}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    Complex{T}(cis(π*(x^2*(a-αx) + y^2*(a-αy))/(b*λ))/λ)
end

function collins_d_chirp(x::T, y::T, λ::T, αx::Tp, αy::Tp, a::Tp, b::Tp,
                         d::Tp) where {T <: Real, Tp <: Real}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    Complex{T}(cis(π*(x^2*αx*(d*αx-1) + y^2*αy*(d*αy-1))/(b*λ)))
end

function collins_convolution_kernel(x::T, y::T, λ::T, αx::Tp, αy::Tp, a::Tp, b::Tp,
                                    d::Tp) where {T <: Real, Tp <: Real}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    Complex{T}(cis(π*(x^2*αx + y^2*αy)/(b*λ)))
end

function collins_tilted_a_chirp(x::T, y::T, λ::T, θx::T, θy::T, track_tilts::Bool,
                                αx::Tp, αy::Tp, a::Tp, b::Tp,
                                d::Tp) where {T <: Real, Tp <: Real}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    f0x, f0y = sin(θx)/λ, sin(θy)/λ
    lin_phase = track_tilts ? Complex{Tp}(1) : cis(Tp(2)*π*(x*f0x+y*f0y))
    Complex{T}(cis(π*(x^2*(a-αx) + y^2*(a-αy))/(b*λ))*lin_phase/λ)
end

function collins_tilted_d_chirp(x::T, y::T, λ::T, θx::T, θy::T,
                                track_tilts::Bool, αx::Tp, αy::Tp, a::Tp, b::Tp,
                                d::Tp) where {T <: Real, Tp <: Real}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    f0x, f0y = sin(θx)/λ, sin(θy)/λ
    lin_phase = track_tilts ? Complex{Tp}(1) : cis(-Tp(2)*π*(x*f0x+y*f0y))
    Complex{T}(cis(π*(x^2*αx*(d*αx-1) + y^2*αy*(d*αy-1))/(b*λ))*lin_phase)
end

function collins_tilted_convolution_kernel(x::T, y::T, λ::T, ::T, ::T, ::Bool,
                                           αx::Tp, αy::Tp, a::Tp, b::Tp,
                                           d::Tp) where {T <: Real, Tp <: Real}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    Complex{T}(cis(π*(x^2*αx + y^2*αy)/(b*λ)))
end

function collins_adjust_tilt(x::T, y::T, λ::T, θx::T, θy::T, a::Tp, ::Tp, d::Tp,
                             ::Type{<:Forward}) where {T <: Real, Tp <: Real}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    f0x, f0y = sin(θx)/λ, sin(θy)/λ
    Complex{T}(cis(-Tp(2)*(d-1)*π*(x*f0x+y*f0y)))
end

function collins_adjust_tilt(x::T, y::T, λ::T, θx::T, θy::T, a::Tp, ::Tp, d::Tp,
                             ::Type{<:Backward}) where {T <: Real, Tp <: Real}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    f0x, f0y = sin(θx)/λ, sin(θy)/λ
    Complex{T}(cis(-Tp(2)*(a-1)*π*(x*f0x+y*f0y)))
end

struct CollinsKernel{K, Nd, V, P, U} <: AbstractKernel{K, V}
    a_chirp::ChirpKernel{K, V}
    d_chirp::ChirpKernel{K, V}
    convolution_kernel::ConvolutionKernel{K, Nd, V, P, U}
end

struct CollinsKernelProp{M, K, T, Tp, Nd} <: AbstractPropagator{M, K, T}
    kernel::K
    αs::NTuple{Nd, Tp}
    ds::NTuple{Nd, Tp}
    ds′::NTuple{Nd, Tp}
    abd::Tuple{Tp, Tp, Tp}
    track_tilts::Bool
    adjust_tilts::Bool
    nrm_fwd::Complex{Tp}
    nrm_bwd::Complex{Tp}

    function CollinsKernelProp(u::ScalarField{U, Nd},
                               ds::NTuple{Nd, Real},
                               ds′::NTuple{Nd, Real},
                               abd::Tuple{Real, Real, Real};
                               use_cache::Bool = true,
                               track_tilts::Bool = false,
                               adjust_tilts::Bool = true,
                               double_precision_kernel::Bool
                               = use_cache) where {T, U <: AbstractArray{Complex{T}}, Nd}
        ns = size(u)[1:Nd]
        cache_size = use_cache ? prod(size(u)[(Nd + 1):end]) : 0
        a_chirp = ChirpKernel(u.electric, ns, ds, cache_size)
        d_chirp = ChirpKernel(u.electric, ns, ds, cache_size)
        conv_kernel = ConvolutionKernel(u.electric, ns, ds, cache_size)
        kernel = CollinsKernel(a_chirp, d_chirp, conv_kernel)
        Tp = double_precision_kernel ? Float64 : T
        αs = Tuple([Tp(dx′/dx) for (dx, dx′) in zip(ds, ds′)])
        _, b = abd
        nrm_fwd = Complex{Tp}(prod(ds ./ sqrt(im*b)))
        nrm_bwd = Complex{Tp}(prod(ds′ ./ sqrt(-im*b)))
        new{Static, typeof(kernel), T, Tp, Nd}(kernel, αs, ds, ds′, Tp.(abd), track_tilts,
                                               adjust_tilts, nrm_fwd, nrm_bwd)
    end

    function CollinsKernelProp(u::ScalarField{U, Nd},
                               ds′::NTuple{Nd, Real},
                               abd::Tuple{Real, Real, Real};
                               use_cache::Bool = true,
                               track_tilts::Bool = false,
                               adjust_tilts::Bool = true,
                               double_precision_kernel::Bool = use_cache) where {U, Nd}
        CollinsKernelProp(u, Tuple(u.ds), ds′, abd; use_cache, track_tilts, adjust_tilts,
                          double_precision_kernel)
    end
end

Functors.@functor CollinsKernelProp ()

function get_kernels(p::CollinsKernelProp)
    (p.kernel.a_chirp, p.kernel.d_chirp, p.kernel.convolution_kernel)
end

function build_kernel_key_args(p::CollinsKernelProp, u::ScalarField)
    if is_on_axis(u)
        (select_lambdas(u),)
    else
        (select_lambdas(u), select_tilts(u)...)
    end
end

function build_kernel_args(p::CollinsKernelProp, u::ScalarField)
    if is_on_axis(u)
        (p.αs..., p.abd...)
    else
        (p.track_tilts, p.αs..., p.abd...)
    end
end

function apply_collins_first_chirp!(u_tmp, apply_a!, apply_d!, on_axis, ::Type{Forward})
    if on_axis
        apply_a!(u_tmp, collins_a_chirp)
    else
        apply_a!(u_tmp, collins_tilted_a_chirp)
    end
end

function apply_collins_first_chirp!(u_tmp, apply_a!, apply_d!, on_axis, ::Type{Backward})
    if on_axis
        apply_d!(u_tmp, collins_d_chirp)
    else
        apply_d!(u_tmp, collins_tilted_d_chirp)
    end
end

function apply_collins_last_chirp!(u_tmp, apply_a!, apply_d!, on_axis, direction)
    apply_collins_first_chirp!(u_tmp, apply_a!, apply_d!, on_axis, conj(direction))
end

function normalize_collins!(u::AbstractArray, p::CollinsKernelProp, ::Type{Forward})
    u .*= p.nrm_fwd
end

function normalize_collins!(u::AbstractArray, p::CollinsKernelProp, ::Type{Backward})
    u .*= p.nrm_bwd
end

function _propagate_core!(apply_kernel_fns::F,
                          u::ScalarField,
                          p::CollinsKernelProp,
                          direction::Type{<:Direction}) where {F}
    apply_a_chirp!, apply_d_chirp!, apply_kernel_fn! = apply_kernel_fns
    p_f = p.kernel.convolution_kernel.p_f
    u_tmp = p.kernel.convolution_kernel.u_plan
    u_tmp .= 0
    u_view = view(u_tmp, axes(u.electric)...)
    copyto!(u_view, u.electric)
    on_axis = is_on_axis(u)
    apply_collins_first_chirp!(u_tmp, apply_a_chirp!, apply_d_chirp!, on_axis, direction)
    p_f.ft * u_tmp
    if on_axis
        apply_kernel_fn!(u_tmp, collins_convolution_kernel)
    else
        apply_kernel_fn!(u_tmp, collins_tilted_convolution_kernel)
    end
    p_f.ift * u_tmp
    apply_collins_last_chirp!(u_tmp, apply_a_chirp!, apply_d_chirp!, on_axis, direction)
    if !p.track_tilts && p.adjust_tilts
        sv = p.kernel.a_chirp.s_vec
        lambdas = u.lambdas.val
        @. u_tmp *= collins_adjust_tilt(sv..., lambdas, u.tilts.val..., p.abd..., direction)
    end
    copyto!(u.electric, u_view)
    normalize_collins!(u.electric, p, direction)
    if !p.track_tilts && p.adjust_tilts
        a, _, d = p.abd
        if direction == Forward
            foreach(θ -> (@. θ *= d), u.tilts.val)
            foreach(θ -> (@. θ *= d), u.tilts.collection)
        else
            foreach(θ -> (@. θ *= a), u.tilts.val)
            foreach(θ -> (@. θ *= a), u.tilts.collection)
        end
    end
    u
end

function set_ds_out!(p::CollinsKernelProp, u::ScalarField, ::Type{Forward})
    set_field_ds!(u, p.ds′)
end

function set_ds_out!(p::CollinsKernelProp, u::ScalarField, ::Type{Backward})
    set_field_ds!(u, p.ds)
end

struct CollinsProp{M, C} <: AbstractSequence{M}
    optical_components::C

    function CollinsProp(optical_components::C) where {C}
        new{Trainable, C}(optical_components)
    end

    function CollinsProp(u::ScalarField{U, Nd},
                         ds::NTuple{Nd, Real},
                         ds′::NTuple{Nd, Real},
                         abd::Tuple{Real, Real, Real};
                         use_cache::Bool = true,
                         track_tilts::Bool = false,
                         adjust_tilts::Bool = false,
                         double_precision_kernel::Bool = use_cache) where {U, Nd}
        kernel = CollinsKernelProp(u, ds, ds′, abd; use_cache, track_tilts, adjust_tilts,
                                   double_precision_kernel)
        M = get_trainability(kernel)
        if !track_tilts && adjust_tilts
            trainable, buffered = istrainable(kernel), isbuffered(kernel)
            wrapper = OpticalSequence(TiltAnchor(u; trainable, buffered), kernel)
        else
            wrapper = OpticalSequence(kernel)
        end
        optical_components = get_sequence(wrapper)
        C = typeof(optical_components)
        new{M, C}(optical_components)
    end

    function CollinsProp(u::ScalarField{U, Nd},
                         ds′::NTuple{Nd, Real},
                         abd::Tuple{Real, Real, Real};
                         use_cache::Bool = true,
                         track_tilts::Bool = false,
                         adjust_tilts::Bool = false,
                         double_precision_kernel::Bool = use_cache) where {U, Nd}
        CollinsProp(u, Tuple(u.ds), ds′, abd; use_cache, track_tilts, adjust_tilts,
                    double_precision_kernel)
    end
end

Functors.@functor CollinsProp (optical_components,)

get_sequence(p::CollinsProp) = p.optical_components
