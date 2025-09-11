function tilted_as_kernel(fx::T, fy::T, λ::T, θx::T, θy::T, n0::Tp, z::Tp,
        filter::H, z_pos::Val{true}) where {T <: Real, Tp <: Real, H}
    fx, fy, λ = Tp(fx), Tp(fy), Tp(λ)/n0
    θx, θy = Tp(θx), Tp(θy)
    f² = complex(inv(λ)^2)
    f0x, f0y = sin(θx)/λ, sin(θy)/λ
    v = isnothing(filter) ? Tp(1) : Tp(filter(fx+f0x, fy+f0y))
    Complex{T}(cis(Tp(2)*π*z*sqrt(f²-(fx+f0x)^2-(fy+f0y)^2)) * v)
end

function tilted_as_kernel(fx::T, fy::T, λ::T, θx::T, θy::T, n0::Tp, z::Tp,
        filter::H, z_pos::Val{false}) where {T <: Real, Tp <: Real, H}
    fx, fy, λ = Tp(fx), Tp(fy), Tp(λ)/n0
    θx, θy = Tp(θx), Tp(θy)
    f² = complex(inv(λ)^2)
    v = isnothing(filter) ? Tp(1) : Tp(filter(fx, fy))
    f0x, f0y = sin(θx)/λ, sin(θy)/λ
    Complex{T}(conj(cis(Tp(2)*π*(-z)*sqrt(f²-(fx+f0x)^2-(fy+f0y)^2)) * v))
end

function tilted_as_kernel(fx::T, λ::T, θx::T, n0::Tp, z::Tp, filter::H,
        z_pos::Val{true}) where {T <: Real, Tp <: Real, H}
    fx, λ, θx = Tp(fx), Tp(λ)/n0, Tp(θx)
    f² = complex(inv(λ^2))
    v = isnothing(filter) ? Tp(1) : Tp(filter(fx))
    f0x = sin(θx)/λ
    Complex{T}(cis(Tp(2)*π*z*sqrt(f²-(fx+f0x)^2)) * v)
end

function tilted_as_kernel(fx::T, λ::T, θx::T, n0::Tp, z::Tp, filter::H,
        z_pos::Val{false}) where {T <: Real, Tp <: Real, H}
    fx, λ, θx = Tp(fx), Tp(λ)/n0, Tp(θx)
    f² = complex(inv(λ^2))
    v = isnothing(filter) ? Tp(1) : Tp(filter(fx))
    f0x = sin(θx)/λ
    Complex{T}(conj(cis(Tp(2)*π*(-z)*sqrt(f²-(fx+f0x)^2)) * v))
end

struct TiltedASProp{M, K, T, Tp, H} <: AbstractPropagator{M, K, T}
    kernel::K
    n0::Tp
    z::Tp
    filter::H

    function TiltedASProp(u::ScalarField{U, Nd},
            ds::NTuple{Nd, Real},
            z::Real;
            use_cache::Bool = true,
            n0::Real = 1,
            filter::H = nothing,
            double_precision_kernel::Bool = true
    ) where {N, Nd, T, H, U <: AbstractArray{Complex{T}, N}}
        ns = size(u)[1:Nd]
        Tp = double_precision_kernel ? Float64 : T
        cache_size = use_cache ? prod(size(u)[(Nd + 1):end]) : 0
        kernel = FourierKernel(u.data, ns, ds, cache_size)
        K = typeof(kernel)
        new{Static, K, T, Tp, H}(kernel, Tp(n0), Tp(z), filter)
    end

    function TiltedASProp(u::ScalarField, z::Real;
            use_cache::Bool = true,
            n0::Real = 1,
            filter = nothing,
            double_precision_kernel::Bool = true)
        TiltedASProp(u, u.ds, z; use_cache, n0, filter, double_precision_kernel)
    end
end

Functors.@functor TiltedASProp ()

get_kernels(p::TiltedASProp) = (p.kernel,)

function build_kernel_key_args(p::TiltedASProp, u::ScalarField)
    (select_lambdas(u), select_tilts(u)...)
end

build_kernel_args(p::TiltedASProp) = (p.n0, p.z, p.filter, Val(sign(p.z) > 0))

function _propagate_core!(apply_kernel_fns::F, u::AbstractArray, p::TiltedASProp,
        ::Type{<:Direction}) where {F}
    apply_kernel_fn!, = apply_kernel_fns
    p_f = p.kernel.p_f
    p_f.ft * u
    apply_kernel_fn!(u, tilted_as_kernel)
    p_f.ift * u
end
