using Polynomials
using LinearAlgebra

function get_zR(w0, λ)
    π*w0^2/λ
end

function get_ψG(z, zR)
    atan(z, zR)
end

function get_w(w0, λ, z)
    zR = get_zR(w0, λ)
    w0*sqrt(1+(z/zR)^2)
end

function get_q(w0, λ, z)
    zR = get_zR(w0, λ)
    z - im*zR
end

function gaussian_normalisation_constant(w0)
    sqrt(sqrt(2/π)/w0)
end

function hg_normalisation_constant(m)
    1 / sqrt(2^m*factorial(m))
end

function hg_normalisation_constant(w0, m)
    gaussian_normalisation_constant(w0) * hg_normalisation_constant(m)
end

function lg_normalisation_constant(w0, p, l)
    sqrt(2*factorial(p) / (π*factorial(p + abs(l))))/w0
end

struct Gaussian1D{T, P <: Union{Nothing, <:NamedTuple}} <: Mode{1}
    w0::T
    C::Complex{T}
    data::P

    function Gaussian1D(w0::Real, λ::Real, z::Real; constant_phase = true)
        T = float(eltype(w0))
        q0 = Complex{T}(get_q(w0, λ, 0))
        qz = Complex{T}(get_q(w0, λ, z))
        wz = Complex{T}(get_w(w0, λ, z))
        k = T(2π/λ)
        eikz = constant_phase ? Complex{T}(exp(im*k*z)) : Complex{T}(1)
        C = gaussian_normalisation_constant(w0) * sqrt(q0 / qz)
        data = (λ = λ, z = z, wz = wz, qz = qz, eikz = eikz, e_arg = im*k/(2*qz))
        new{T, typeof(data)}(w0, C, data)
    end

    function Gaussian1D(w0::Real)
        T = float(eltype(w0))
        C = Complex{T}(gaussian_normalisation_constant(w0))
        new{T, Nothing}(w0, C, nothing)
    end
end

function Base.eltype(m::Gaussian1D{T}) where {T}
    Complex{T}
end

function eval_exp_arg(m::Gaussian1D{<:Real, Nothing}, x)
    -(x/m.w0)^2
end

function eval_exp_arg(m::Gaussian1D{<:Real, <:NamedTuple}, x)
    d = m.data
    d.e_arg * x^2
end

function eval_mode(m::Gaussian1D{<:Real, Nothing}, x)
    m.C * exp(eval_exp_arg(m, x))
end

function eval_mode(m::Gaussian1D{<:Real, <:NamedTuple}, x)
    d = m.data
    sqrt(m.C) * exp(eval_exp_arg(m, x)) * d.eikz
end

struct Gaussian{G <: Gaussian1D} <: Mode{2}
    gx::G
    gy::G

    function Gaussian(w0x::Real, w0y::Real, λ::Real, z::Real; constant_phase = true)
        T = float(promote_type(typeof(w0x), typeof(w0y)))
        gx = Gaussian1D(T(w0x), λ, z; constant_phase = constant_phase)
        gy = Gaussian1D(T(w0y), λ, z; constant_phase = false)
        # We don't want to account for constant phase twice
        new{typeof(gx)}(gx, gy)
    end

    function Gaussian(w0x::Real, w0y::Real)
        gx = Gaussian1D(w0x)
        gy = Gaussian1D(w0y)
        new{typeof(gx)}(gx, gy)
    end

    function Gaussian(w0::Real, λ::Real, z::Real; constant_phase = true)
        Gaussian(w0, w0, λ, z; constant_phase = constant_phase)
    end

    function Gaussian(w0::Real)
        Gaussian(w0, w0)
    end
end

function Base.eltype(m::Gaussian)
    promote_type(eltype(m.gx), eltype(m.gy))
end

function eval_constant_phase(m::Gaussian1D{T, Nothing}) where {T <: Real}
    Complex{T}(1)
end

function eval_constant_phase(m::Gaussian1D{<:Real, <:NamedTuple})
    m.data.eikz
end

function eval_mode(m::Gaussian, x, y)
    Cx, Cy = m.gx.C, m.gy.C
    eikz = eval_constant_phase(m.gx)
    Cx * Cy * exp(eval_exp_arg(m.gx, x) + eval_exp_arg(m.gy, y)) * eikz
end

function hermite_polynomial(n::Integer)
    if n < 0
        throw(DomainError())
    elseif n == 0
        Polynomial([1])
    elseif n == 1
        Polynomial([0, 2])
    else
        2*Polynomial([0, 1])*hermite_polynomial(n-1)-2*(n-1)*hermite_polynomial(n-2)
    end
end

struct HermiteGaussian1D{T, G <: Gaussian1D{T}, P} <: Mode{1}
    g::G
    C::Complex{T}
    hn::P
    n::Int

    function HermiteGaussian1D(w0::Real, n::Integer, λ::Real, z::Real;
            constant_phase = true)
        T = float(eltype(w0))
        g = Gaussian1D(w0, λ, z; constant_phase = constant_phase)
        hn = hermite_polynomial(n)
        qz = g.data.qz
        C = hg_normalisation_constant(n)*sqrt((-conj(qz)/qz)^n)
        new{T, typeof(g), typeof(hn)}(g, C, hn, n)
    end

    function HermiteGaussian1D(w0::Real, n::Integer)
        T = float(eltype(w0))
        g = Gaussian1D(w0)
        hn = hermite_polynomial(n)
        C = hg_normalisation_constant(n)
        new{T, typeof(g), typeof(hn)}(g, C, hn, n)
    end
end

function Base.eltype(m::HermiteGaussian1D{T}) where {T}
    Complex{T}
end

function eval_wz(m::HermiteGaussian1D{T, <:Gaussian1D{T, Nothing}}) where {T}
    m.g.w0
end

function eval_wz(m::HermiteGaussian1D{T, <:Gaussian1D{T, <:NamedTuple}}) where {T}
    m.g.data.wz
end

function eval_mode(m::HermiteGaussian1D, x)
    wz = eval_wz(m)
    m.C * m.hn(sqrt(2)*x/wz) * eval_mode(m.g, x)
end

struct HermiteGaussian{G <: HermiteGaussian1D} <: Mode{2}
    hgx::G
    hgy::G

    function HermiteGaussian(
            w0x::Real, w0y::Real, m::Integer, n::Integer, λ::Real, z::Real;
            constant_phase = true)
        T = float(promote_type(typeof(w0x), typeof(w0y)))
        hgx = HermiteGaussian1D(T(w0x), m, λ, z; constant_phase = constant_phase)
        hgy = HermiteGaussian1D(T(w0y), n, λ, z; constant_phase = false)
        # We don't want to account for constant phase twice
        new{typeof(hgx)}(hgx, hgy)
    end

    function HermiteGaussian(w0x::Real, w0y::Real, m::Integer, n::Integer)
        hgx = HermiteGaussian1D(w0x, m)
        hgy = HermiteGaussian1D(w0y, n)
        new{typeof(hgx)}(hgx, hgy)
    end

    function HermiteGaussian(w0::Real, m::Integer, n::Integer, λ::Real, z::Real;
            constant_phase = true)
        HermiteGaussian(w0, w0, m, n, λ, z; constant_phase = constant_phase)
    end

    function HermiteGaussian(w0::Real, m::Integer, n::Integer)
        HermiteGaussian(w0, w0, m, n)
    end
end

function Base.eltype(m::HermiteGaussian)
    promote_type(eltype(m.hgx), eltype(m.hgy))
end

function eval_mode(m::HermiteGaussian, x, y)
    mx = m.hgx
    my = m.hgy
    wz_x = eval_wz(mx)
    wz_y = eval_wz(my)
    C = mx.C * mx.g.C * my.C * my.g.C
    eikz = eval_constant_phase(mx.g)
    (C * mx.hn(sqrt(2)*x/wz_x) * my.hn(sqrt(2)*y/wz_y)
     * exp(eval_exp_arg(mx.g, x) + eval_exp_arg(my.g, y)) * eikz)
end

function hermite_gaussian_groups(w0, n_groups::Int)
    @assert n_groups >= 0
    l = HermiteGaussian[]
    for m in 0:(n_groups - 1)
        for n in 0:(n_groups - m - 1)
            push!(l, HermiteGaussian(w0, m, n))
        end
    end
    l
end

function laguerre_polynomial(p::Integer, l::Integer)
    if p < 0
        throw(DomainError())
    elseif p == 0
        Polynomial([1])
    elseif p == 1
        Polynomial([1+l, -1])
    else
        ((2*p+l-1-Polynomial([0, 1]))*laguerre_polynomial(p-1, l) -
         (p+l-1)*laguerre_polynomial(p-2, l)) / p
    end
end

struct LaguerreGaussian{T, P <: Union{Nothing, <:NamedTuple}, K, L} <: Mode{2}
    w0::T
    C::Complex{T}
    lp::L
    order::@NamedTuple{p::Int, l::Int}
    data::P
    kind::K

    function LaguerreGaussian(w0::Real, p::Integer, l::Integer, λ::Real, z::Real;
            constant_phase = true, kind = :vortex)
        @assert p >= 0
        T = float(eltype(w0))
        q0 = Complex{T}(get_q(w0, λ, 0))
        qz = Complex{T}(get_q(w0, λ, z))
        wz = Complex{T}(get_w(w0, λ, z))
        k = T(2π/λ)
        eikz = constant_phase ? Complex{T}(exp(im*k*z)) : Complex{T}(1)
        C = lg_normalisation_constant(w0, p, l) * (q0/qz) * (-conj(qz)/qz)^(p+abs(l)/2)
        lp = laguerre_polynomial(p, abs(l))
        data = (λ = λ, z = z, wz = wz, eikz = eikz, e_arg = im*k/(2*qz))
        kind = parse_kind(kind)
        P = typeof(data)
        L = typeof(lp)
        K = typeof(kind)
        new{T, P, K, L}(w0, C, lp, (p = p, l = l), data, kind)
    end

    function LaguerreGaussian(w0::Real, p::Integer, l::Integer; kind = :vortex)
        T = float(eltype(w0))
        C = Complex{T}(gaussian_normalisation_constant(w0))
        lp = laguerre_polynomial(p, abs(l))
        kind = parse_kind(kind)
        L = typeof(lp)
        K = typeof(kind)
        new{T, Nothing, K, L}(w0, C, lp, (p = p, l = l), nothing, kind)
    end
end

function Base.eltype(m::LaguerreGaussian{T}) where {T}
    Complex{T}
end

function eval_lg_arg(m::LaguerreGaussian{T, P, Vortex}, x, y) where {T, P}
    l = m.order.l
    exp(im*l*atan(y, x))
end

function eval_lg_arg(m::LaguerreGaussian{T, P, Even}, x, y) where {T, P}
    l = m.order.l
    sqrt(2)*cos(l*atan(y, x))
end

function eval_lg_arg(m::LaguerreGaussian{T, P, Odd}, x, y) where {T, P}
    l = m.order.l
    sqrt(2)*sin(l*atan(y, x))
end

function eval_mode(m::LaguerreGaussian{T, Nothing}, x, y) where {T}
    l = m.order.l
    r2 = (x^2 + y^2) / m.w0^2
    rl = (sqrt(2)*sqrt(r2))^abs(l)
    m.C * rl * m.lp(2*r2) * exp(-r2) * eval_lg_arg(m, x, y)
end

function eval_mode(m::LaguerreGaussian{T, <:NamedTuple}, x, y) where {T}
    l = m.order.l
    wz = m.data.wz
    r2 = x^2 + y^2
    rl = (sqrt(2)*sqrt(r2)/wz)^abs(l)
    m.C * rl * m.lp(2*r2/wz^2) * exp(m.data.e_arg*r2) * eval_lg_arg(m, x, y) * m.data.eikz
end
