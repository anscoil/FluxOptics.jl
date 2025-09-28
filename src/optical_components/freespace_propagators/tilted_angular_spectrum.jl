function tilted_as_kernel(fx::T, fy::T, λ::T, θx::T, θy::T, n0::Tp, z::Tp, filter::H,
                          z_pos::Val{true}) where {T <: Real, Tp <: Real, H}
    fx, fy, λ = Tp(fx), Tp(fy), Tp(λ)/n0
    θx, θy = Tp(θx), Tp(θy)
    f² = complex(inv(λ)^2)
    f0x, f0y = sin(θx)/λ, sin(θy)/λ
    v = isnothing(filter) ? Complex{Tp}(1) : Complex{Tp}(filter(fx+f0x, fy+f0y))
    Complex{T}(cis(Tp(2)*π*z*sqrt(f²-(fx+f0x)^2-(fy+f0y)^2)) * v)
end

function tilted_as_kernel(fx::T, fy::T, λ::T, θx::T, θy::T, n0::Tp, z::Tp, filter::H,
                          z_pos::Val{false}) where {T <: Real, Tp <: Real, H}
    fx, fy, λ = Tp(fx), Tp(fy), Tp(λ)/n0
    θx, θy = Tp(θx), Tp(θy)
    f² = complex(inv(λ)^2)
    f0x, f0y = sin(θx)/λ, sin(θy)/λ
    v = isnothing(filter) ? Complex{Tp}(1) : Complex{Tp}(filter(fx, fy))
    Complex{T}(conj(cis(Tp(2)*π*(-z)*sqrt(f²-(fx+f0x)^2-(fy+f0y)^2)) * v))
end

function tilted_as_kernel(fx::T, λ::T, θx::T, n0::Tp, z::Tp, filter::H,
                          z_pos::Val{true}) where {T <: Real, Tp <: Real, H}
    fx, λ, θx = Tp(fx), Tp(λ)/n0, Tp(θx)
    f² = complex(inv(λ^2))
    f0x = sin(θx)/λ
    v = isnothing(filter) ? Complex{Tp}(1) : Complex{Tp}(filter(fx+f0x))
    Complex{T}(cis(Tp(2)*π*z*sqrt(f²-(fx+f0x)^2)) * v)
end

function tilted_as_kernel(fx::T, λ::T, θx::T, n0::Tp, z::Tp, filter::H,
                          z_pos::Val{false}) where {T <: Real, Tp <: Real, H}
    fx, λ, θx = Tp(fx), Tp(λ)/n0, Tp(θx)
    f² = complex(inv(λ^2))
    f0x = sin(θx)/λ
    v = isnothing(filter) ? Complex{Tp}(1) : Complex{Tp}(filter(fx+f0x))
    Complex{T}(conj(cis(Tp(2)*π*(-z)*sqrt(f²-(fx+f0x)^2)) * v))
end

struct TiltedASKernel{M, K, T, Tp, H} <: AbstractPropagator{M, K, T}
    kernel::K
    n0::Tp
    z::Tp
    filter::H

    function TiltedASKernel(u::ScalarField{U, Nd},
                            ds::NTuple{Nd, Real},
                            z::Real;
                            use_cache::Bool = true,
                            n0::Real = 1,
                            filter::H = nothing,
                            double_precision_kernel::Bool = use_cache) where {N, Nd, T, H,
                                                                              U <:
                                                                              AbstractArray{Complex{T},
                                                                                            N}}
        ns = size(u)[1:Nd]
        Tp = double_precision_kernel ? Float64 : T
        cache_size = use_cache ? prod(size(u)[(Nd + 1):end]) : 0
        kernel = FourierKernel(u.electric, ns, ds, cache_size)
        K = typeof(kernel)
        new{Static, K, T, Tp, H}(kernel, Tp(n0), Tp(z), filter)
    end

    function TiltedASKernel(u::ScalarField,
                            z::Real;
                            use_cache::Bool = true,
                            n0::Real = 1,
                            filter = nothing,
                            double_precision_kernel::Bool = use_cache)
        TiltedASKernel(u, u.ds, z; use_cache, n0, filter, double_precision_kernel)
    end
end

Functors.@functor TiltedASKernel ()

get_kernels(p::TiltedASKernel) = (p.kernel,)

function build_kernel_key_args(p::TiltedASKernel, u::ScalarField)
    (select_lambdas(u), select_tilts(u)...)
end

build_kernel_args(p::TiltedASKernel) = (p.n0, p.z, p.filter, Val(sign(p.z) > 0))

function _propagate_core!(apply_kernel_fns::F,
                          u::AbstractArray,
                          p::TiltedASKernel,
                          ::Type{<:Direction}) where {F}
    apply_kernel_fn!, = apply_kernel_fns
    apply_kernel_fn!(u, tilted_as_kernel)
    u
end

struct TiltedASProp{M, C} <: AbstractSequence{M}
    optical_components::C

    function TiltedASProp(u::ScalarField{U, Nd},
                          ds::NTuple{Nd, Real},
                          z::Real;
                          use_cache::Bool = true,
                          n0::Real = 1,
                          filter = nothing,
                          double_precision_kernel::Bool = use_cache) where {U, Nd}
        kernel = TiltedASKernel(u, ds, z; use_cache, n0, filter, double_precision_kernel)
        wrapper = FourierWrapper(kernel.kernel.p_f, kernel)
        M = get_trainability(wrapper)
        optical_components = get_sequence(wrapper)
        C = typeof(optical_components)
        new{M, C}(optical_components)
    end

    function TiltedASProp(u::ScalarField,
                          z::Real;
                          use_cache::Bool = true,
                          n0::Real = 1,
                          filter = nothing,
                          double_precision_kernel::Bool = use_cache)
        TiltedASProp(u, u.ds, z; use_cache, n0, filter, double_precision_kernel)
    end
end

Functors.@functor TiltedASProp (optical_components,)

get_sequence(p::TiltedASProp) = p.optical_components
