function tilted_as_kernel(fx::T, fy::T, λ::T, θx::T, θy::T, n0::Tp, z::Tp,
        filter::H, z_pos::Val{true}) where {T <: Real, Tp <: Real, H}
    fx, fy, λ = Tp(fx), Tp(fy), Tp(λ)/n0
    θx, θy = Tp(θx), Tp(θy)
    f² = complex(inv(λ)^2)
    v = isnothing(filter) ? Tp(1) : Tp(filter(fx, fy))
    f0x, f0y = sin(θx)/λ, sin(θy)/λ
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

function parse_tilt_vectors(u::U,
        θs::NTuple{Nd, Union{Real, AbstractVector{<:Real}}}
) where {Nd, U <: AbstractArray{<:Complex}}
    shape = ntuple(k -> k <= Nd ? 1 : size(u, k), ndims(u))
    V = adapt_dim(U, 1, real)
    map(θ -> isa(θ, Real) ? V([θ]) : reshape(V(θ), shape), θs)
end

function collect_tilt_vectors(u::U,
        θs::NTuple{Nd, Union{Real, AbstractVector{<:Real}}}
) where {Nd, T <: Real, U <: AbstractArray{Complex{T}}}
    shape = ntuple(k -> k <= Nd ? 1 : size(u, k), ndims(u))
    map(θ -> isa(θ, Real) ? fill(T(θ), shape) : reshape(Array{T}(θ), shape), θs)
end

struct TiltedASProp{M, K, T, Tp, O, Oc, H} <: AbstractPropagator{M, K, T}
    kernel::K
    n0::Tp
    z::Tp
    θs::O
    θs_collection::Oc
    filter::H

    function TiltedASProp(u::AbstractArray{Complex{T}, N},
            ds::NTuple{Nd, Real},
            z::Real,
            θs::NTuple{Nd, Union{Real, AbstractVector{<:Real}}},
            λ::Real;
            n0::Real = 1,
            filter::H = nothing,
            double_precision_kernel::Bool = true) where {N, Nd, T, H}
        @assert N >= Nd
        ns = size(u)[1:Nd]
        Tp = double_precision_kernel ? Float64 : T
        θs = parse_tilt_vectors(u, θs)
        kernel = FourierKernel(u, ns, ds, 1, N)
        kernel_key = hash(T(λ))
        fill_kernel_cache(kernel, kernel_key, tilted_as_kernel,
            (T(λ), θs..., Tp(n0), Tp(z), filter, Val(sign(z > 0))))
        K = typeof(kernel)
        O = typeof(θs)
        new{Static, K, T, Tp, O, Nothing, H}(kernel, Tp(n0), Tp(z), θs, nothing, filter)
    end

    function TiltedASProp(u::ScalarField{U, Nd},
            ds::NTuple{Nd, Real},
            z::Real,
            θs::NTuple{Nd, Union{Real, AbstractVector{<:Real}}},
            use_cache::Bool = false;
            n0::Real = 1,
            filter::H = nothing,
            paraxial::Bool = false,
            double_precision_kernel::Bool = true
    ) where {N, Nd, T, H, U <: AbstractArray{Complex{T}, N}}
        ns = size(u)[1:Nd]
        Tp = double_precision_kernel ? Float64 : T
        if isa(u.lambdas, Real)
            cache_size = use_cache ? 1 : 0
            kernel = FourierKernel(u.data, ns, ds, cache_size, N)
        else
            cache_size = use_cache ? prod(size(u)[(Nd + 1):end]) : 0
            kernel = FourierKernel(u.data, ns, ds, cache_size)
        end
        θs_collection = use_cache ? collect_tilt_vectors(u.data, θs) : nothing
        θs = parse_tilt_vectors(u.data, θs)
        K = typeof(kernel)
        O = typeof(θs)
        Oc = typeof(θs_collection)
        new{Static, K, T, Tp, O, Oc, H}(kernel, Tp(n0), Tp(z), θs, θs_collection, filter)
    end

    function TiltedASProp(u::ScalarField, z::Real,
            θs::NTuple{Nd, Union{Real, AbstractVector{<:Real}}},
            use_cache::Bool = false;
            n0::Real = 1,
            filter = nothing,
            paraxial::Bool = false,
            double_precision_kernel::Bool = true) where {Nd}
        TiltedASProp(u, u.ds, z, θs, use_cache; n0, filter, double_precision_kernel)
    end
end

Functors.@functor TiltedASProp ()

get_kernels(p::TiltedASProp) = (p.kernel,)

build_kernel_args(p::TiltedASProp) = (p.θs..., p.n0, p.z, p.filter, Val(sign(p.z) > 0))

get_kernel_extra_key_params(p::TiltedASProp) = p.θs_collection

function _propagate_core!(apply_kernel_fns::F, u::AbstractArray, p::TiltedASProp,
        ::Type{<:Direction}) where {F}
    apply_kernel_fn!, = apply_kernel_fns
    p_f = p.kernel.p_f
    p_f.ft * u
    apply_kernel_fn!(u, tilted_as_kernel)
    p_f.ift * u
end
