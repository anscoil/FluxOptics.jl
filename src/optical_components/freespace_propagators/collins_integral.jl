# Stuart A. Collins, "Lens-System Diffraction Integral Written in
# Terms of Matrix Optics*," J. Opt. Soc. Am. 60, 1168-1177 (1970)
# https://doi.org/10.1364/JOSA.60.001168

function collins_a_chirp(x::T, y::T, λ::T, αx::Tp, αy::Tp, a::Tp,
                         b::Tp) where {T <: Real, Tp <: Real}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    Complex{T}(cis(π*(x^2*(a-αx) + y^2*(a-αy))/(b*λ))/λ)
end

function collins_d_chirp(x::T, y::T, λ::T, αx::Tp, αy::Tp, d::Tp,
                         b::Tp) where {T <: Real, Tp <: Real}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    Complex{T}(cis(π*(x^2*αx*(d*αx-1) + y^2*αy*(d*αy-1))/(b*λ)))
end

function collins_convolution_kernel(x::T, y::T, λ::T, αx::Tp, αy::Tp,
                                    b::Tp) where {T <: Real, Tp <: Real}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    Complex{T}(cis(π*(x^2*αx + y^2*αy)/(b*λ)))
end

function collins_tilted_a_chirp(x::T, y::T, λ::T, θx::T, θy::T, track_tilts::Bool,
                                αx::Tp, αy::Tp, a::Tp, b::Tp) where {T <: Real, Tp <: Real}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    f0x, f0y = sin(θx)/λ, sin(θy)/λ
    lin_phase = track_tilts ? Complex{Tp}(1) : cis(Tp(2)*π*(x*f0x+y*f0y))
    Complex{T}(cis(π*(x^2*(a-αx) + y^2*(a-αy))/(b*λ))*lin_phase/λ)
end

function collins_tilted_d_chirp(x::T, y::T, λ::T, θx::T, θy::T, track_tilts::Bool,
                                αx::Tp, αy::Tp, d::Tp, b::Tp) where {T <: Real, Tp <: Real}
    x, y, λ = Tp(x), Tp(y), Tp(λ)
    f0x, f0y = sin(θx)/λ, sin(θy)/λ
    lin_phase = track_tilts ? Complex{Tp}(1) : cis(-Tp(2)*π*(x*f0x+y*f0y))
    Complex{T}(cis(π*(x^2*αx*(d*αx-1) + y^2*αy*(d*αy-1))/(b*λ))*lin_phase)
end

# function collins_adjust_tilt(x::T, y::T, λ::T, θx::T, θy::T, d::Tp,
#                              ::Type{<:Forward}) where {T <: Real, Tp <: Real}
#     x, y, λ = Tp(x), Tp(y), Tp(λ)
#     f0x, f0y = sin(θx)/λ, sin(θy)/λ
#     Complex{T}(cis(-Tp(2)*(d-1)*π*(x*f0x+y*f0y)))
# end

# function collins_adjust_tilt(x::T, y::T, λ::T, θx::T, θy::T, a::Tp,
#                              ::Type{<:Backward}) where {T <: Real, Tp <: Real}
#     x, y, λ = Tp(x), Tp(y), Tp(λ)
#     f0x, f0y = sin(θx)/λ, sin(θy)/λ
#     Complex{T}(cis(-Tp(2)*(a-1)*π*(x*f0x+y*f0y)))
# end

struct CollinsAChirp{M, K, T, Tp, Nd} <: AbstractPropagator{M, K, T}
    kernel::K
    αs::NTuple{Nd, Tp}
    ds::NTuple{Nd, Tp}
    a::Tp
    b::Tp
    track_tilts::Bool

    function CollinsAChirp(u::ScalarField{U, Nd},
                           ds::NTuple{Nd, Real},
                           ds′::NTuple{Nd, Real},
                           a::Real, b::Real;
                           use_cache::Bool = true,
                           track_tilts::Bool = false,
                           double_precision_kernel::Bool
                           = use_cache) where {T, U <: AbstractArray{Complex{T}}, Nd}
        ns = size(u)[1:Nd]
        cache_size = use_cache ? prod(size(u)[(Nd + 1):end]) : 0
        kernel = ChirpKernel(u.electric, ns, ds, cache_size)
        Tp = double_precision_kernel ? Float64 : T
        αs = Tuple([Tp(dx′/dx) for (dx, dx′) in zip(ds, ds′)])
        K = typeof(kernel)
        new{Static, K, T, Tp, Nd}(kernel, αs, ds, a, b, track_tilts)
    end
end

struct CollinsDChirp{M, K, T, Tp, Nd} <: AbstractPropagator{M, K, T}
    kernel::K
    αs::NTuple{Nd, Tp}
    ds′::NTuple{Nd, Tp}
    d::Tp
    b::Tp
    track_tilts::Bool

    function CollinsDChirp(u::ScalarField{U, Nd},
                           ds::NTuple{Nd, Real},
                           ds′::NTuple{Nd, Real},
                           d::Real, b::Real;
                           use_cache::Bool = true,
                           track_tilts::Bool = false,
                           double_precision_kernel::Bool
                           = use_cache) where {T, U <: AbstractArray{Complex{T}}, Nd}
        ns = size(u)[1:Nd]
        cache_size = use_cache ? prod(size(u)[(Nd + 1):end]) : 0
        kernel = ChirpKernel(u.electric, ns, ds, cache_size)
        Tp = double_precision_kernel ? Float64 : T
        αs = Tuple([Tp(dx′/dx) for (dx, dx′) in zip(ds, ds′)])
        K = typeof(kernel)
        new{Static, K, T, Tp, Nd}(kernel, αs, ds′, d, b, track_tilts)
    end
end

struct CollinsConvolution{M, K, T, Tp, Nd} <: AbstractPropagator{M, K, T}
    kernel::K
    αs::NTuple{Nd, Tp}
    b::Tp
    nrm_fwd::Complex{Tp}
    nrm_bwd::Complex{Tp}

    function CollinsConvolution(u::ScalarField{U, Nd},
                                ds::NTuple{Nd, Real},
                                ds′::NTuple{Nd, Real},
                                b::Real;
                                use_cache::Bool = true,
                                double_precision_kernel::Bool
                                = use_cache) where {T, U <: AbstractArray{Complex{T}}, Nd}
        ns = size(u)[1:Nd]
        cache_size = use_cache ? prod(size(u)[(Nd + 1):end]) : 0
        kernel = ConvolutionKernel(u.electric, ns, ds, cache_size)
        Tp = double_precision_kernel ? Float64 : T
        nrm_fwd = Complex{Tp}(prod(ds ./ sqrt(im*b)))
        nrm_bwd = Complex{Tp}(prod(ds′ ./ sqrt(-im*b)))
        αs = Tuple([Tp(dx′/dx) for (dx, dx′) in zip(ds, ds′)])
        K = typeof(kernel)
        new{Static, K, T, Tp, Nd}(kernel, αs, b, nrm_fwd, nrm_bwd)
    end
end

Functors.@functor CollinsAChirp ()
Functors.@functor CollinsDChirp ()
Functors.@functor CollinsConvolution ()

CollinsChirpKernel = Union{CollinsAChirp, CollinsDChirp}

get_kernels(p::CollinsChirpKernel) = (p.kernel,)
get_kernels(p::CollinsConvolution) = (p.kernel,)

function build_kernel_key_args(p::CollinsChirpKernel, u::ScalarField)
    if is_on_axis(u)
        (select_lambdas(u),)
    else
        (select_lambdas(u), select_tilts(u)...)
    end
end

build_kernel_key_args(p::CollinsConvolution, u::ScalarField) = (select_lambdas(u),)

function build_kernel_args(p::CollinsAChirp, u::ScalarField)
    if is_on_axis(u)
        (p.αs..., p.a, p.b)
    else
        (p.track_tilts, p.αs..., p.a, p.b)
    end
end

function build_kernel_args(p::CollinsDChirp, u::ScalarField)
    if is_on_axis(u)
        (p.αs..., p.d, p.b)
    else
        (p.track_tilts, p.αs..., p.d, p.b)
    end
end

build_kernel_args(p::CollinsConvolution, u::ScalarField) = (p.αs..., p.b)

function _propagate_core!(apply_kernel_fns::F,
                          u::ScalarField,
                          p::CollinsAChirp,
                          direction::Type{<:Direction}) where {F}
    apply_chirp!, = apply_kernel_fns
    if is_on_axis(u)
        apply_chirp!(u.electric, collins_a_chirp)
    else
        apply_chirp!(u.electric, collins_tilted_a_chirp)
    end
    if direction == Backward
        set_field_ds!(u, p.ds)
    end
    u
end

function _propagate_core!(apply_kernel_fns::F,
                          u::ScalarField,
                          p::CollinsDChirp,
                          direction::Type{<:Direction}) where {F}
    apply_chirp!, = apply_kernel_fns
    if is_on_axis(u)
        apply_chirp!(u.electric, collins_d_chirp)
    else
        apply_chirp!(u.electric, collins_tilted_d_chirp)
    end
    if direction == Forward
        set_field_ds!(u, p.ds′)
    end
    u
end

function _propagate_core!(apply_kernel_fns::F,
                          u::ScalarField,
                          p::CollinsConvolution,
                          direction::Type{<:Direction}) where {F}
    apply_convolution!, = apply_kernel_fns
    apply_convolution!(u.electric, collins_convolution_kernel)
    if direction == Forward
        @. u.electric *= p.nrm_fwd
    else
        @. u.electric *= p.nrm_bwd
    end
    u
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
                         double_precision_kernel::Bool = use_cache) where {U, Nd}
        a, b, d = abd
        a_chirp = CollinsAChirp(u, ds, ds′, a, b; use_cache, track_tilts,
                                double_precision_kernel)
        d_chirp = CollinsDChirp(u, ds, ds′, d, b; use_cache, track_tilts,
                                double_precision_kernel)
        collins = CollinsConvolution(u, ds, ds′, b; use_cache, double_precision_kernel)
        wrapper = FourierWrapper(collins.kernel.p_f, collins)
        pad_op = PadCropOperator(u, collins.kernel.u_plan; store_ref = true)
        crop_op = adjoint(pad_op)
        optical_components = (a_chirp, pad_op, get_sequence(wrapper)..., crop_op, d_chirp)
        C = typeof(optical_components)
        new{Static, C}(optical_components)
    end

    function CollinsProp(u::ScalarField{U, Nd},
                         ds′::NTuple{Nd, Real},
                         abd::Tuple{Real, Real, Real};
                         use_cache::Bool = true,
                         track_tilts::Bool = false,
                         double_precision_kernel::Bool = use_cache) where {U, Nd}
        CollinsProp(u, Tuple(u.ds), ds′, abd; use_cache, track_tilts,
                    double_precision_kernel)
    end
end

Functors.@functor CollinsProp (optical_components,)

get_sequence(p::CollinsProp) = p.optical_components
