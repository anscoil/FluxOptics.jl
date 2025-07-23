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
    sqrt(2/π)/w0
end

function hg_normalisation_constant(m)
    1 / (2^m*factorial(m))
end

function hg_normalisation_constant(w0, m)
    gaussian_normalisation_constant(w0) * hg_normalisation_constant(m)
end

struct Gaussian1D{T, P <: Union{Nothing, <:NamedTuple}} <: Mode{1}
    w0::T
    C::Complex{T}
    data::P

    function Gaussian1D(w0::T, λ::Real, z::Real; constant_phase = true) where {T <: Real}
        q0 = Complex{T}(get_q(w0, λ, 0))
        qz = Complex{T}(get_q(w0, λ, z))
        wz = Complex{T}(get_w(w0, λ, z))
        k = T(2π/λ)
        eikz = constant_phase ? Complex{T}(exp(im*k*z)) : Complex{T}(1)
        C = gaussian_normalisation_constant(w0) * (q0 / qz)
        data = (λ = λ, z = z, wz = wz, qz = qz, eikz = eikz, e_arg = im*k/(2*qz))
        new{T, typeof(data)}(w0, C, data)
    end

    function Gaussian1D(w0::T) where {T <: Real}
        C = Complex{T}(gaussian_normalisation_constant(w0))
        new{T, Nothing}(w0, C, nothing)
    end
end

function eval_exp_arg(m::Gaussian1D{<:Real, Nothing}, x)
    -(x/m.w0)^2
end

function eval_exp_arg(m::Gaussian1D{<:Real, <:NamedTuple}, x)
    d = m.data
    d.e_arg * x^2
end

function eval_mode(m::Gaussian1D{<:Real, Nothing}, x)
    sqrt(m.C) * exp(eval_exp_arg(m, x))
end

function eval_mode(m::Gaussian1D{<:Real, <:NamedTuple}, x)
    d = m.data
    sqrt(m.C) * exp(eval_exp_arg(m, x)) * d.eikz
end

struct Gaussian{G <: Gaussian1D} <: Mode{2}
    gx::G
    gy::G

    function Gaussian(w0x::Real, w0y::Real, λ::Real, z::Real; constant_phase = true)
        T = promote_type(typeof(w0x), typeof(w0y))
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

function eval_mode(m::Gaussian{<:Gaussian1D{<:Real, Nothing}}, x, y)
    Cx, Cy = m.gx.C, m.gy.C
    sqrt(Cx*Cy) * exp(eval_exp_arg(m.gx, x) + eval_exp_arg(m.gy, y))
end

function eval_mode(m::Gaussian{<:Gaussian1D{<:Real, <:NamedTuple}}, x, y)
    d_x = m.gx.data
    Cx, Cy = m.gx.C, m.gy.C
    sqrt(Cx*Cy) * exp(eval_exp_arg(m.gx, x) + eval_exp_arg(m.gy, y)) * d_x.eikz
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

    function HermiteGaussian1D(w0::T, n::Integer, λ::Real, z::Real;
            constant_phase = true) where {T <: Real}
        g = Gaussian1D(w0, λ, z; constant_phase = constant_phase)
        hn = hermite_polynomial(n)
        qz = g.data.qz
        C = hg_normalisation_constant(n)*(-conj(qz)/qz)^n
        new{T, typeof(g), typeof(hn)}(g, C, hn, n)
    end

    function HermiteGaussian1D(w0::T, n::Integer) where {T <: Real}
        g = Gaussian1D(w0)
        hn = hermite_polynomial(n)
        C = hg_normalisation_constant(n)
        new{T, typeof(g), typeof(hn)}(g, C, hn, n)
    end
end

function eval_mode(m::HermiteGaussian1D{T, <:Gaussian1D{T, Nothing}}, x) where {T}
    sqrt(m.C) * m.hn(sqrt(2)*x/m.g.w0) * eval_mode(m.g, x)
end

function eval_mode(m::HermiteGaussian1D{T, <:Gaussian1D{T, <:NamedTuple}}, x) where {T}
    sqrt(m.C) * m.hn(sqrt(2)*x/m.g.data.wz) * eval_mode(m.g, x)
end

struct HermiteGaussian{G <: HermiteGaussian1D} <: Mode{2}
    hgx::G
    hgy::G

    function HermiteGaussian(
            w0x::Real, w0y::Real, m::Integer, n::Integer, λ::Real, z::Real;
            constant_phase = true)
        T = promote_type(typeof(w0x), typeof(w0y))
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

function eval_mode(m::HermiteGaussian, x, y)
    eval_mode(m.hgx, x) * eval_mode(m.hgy, y)
end

function hermite_gaussian(
        w0x,
        w0y,
        λ,
        z,
        m::Integer,
        n::Integer;
        xc = 0.0,
        yc = 0.0,
        θ0 = 0.0,
        constant_phase = true
)
    k = 2π/λ
    q0x = get_q(w0x, λ, 0)
    q0y = get_q(w0y, λ, 0)
    qzx = get_q(w0x, λ, z)
    qzy = get_q(w0y, λ, z)
    wzx = get_w(w0x, λ, z)
    wzy = get_w(w0y, λ, z)
    zRx = get_zR(w0x, λ)
    zRy = get_zR(w0y, λ)
    Hm = hermite_polynomial(m)
    Hn = hermite_polynomial(n)
    cosθ0 = cos(θ0)
    sinθ0 = sin(θ0)
    eiψG = sqrt((-conj(qzx)/qzx)^m) * sqrt((-conj(qzy)/qzy)^n)
    C = sqrt(sqrt(2/π)/(2^m*factorial(m)*w0x)*sqrt(2/π)/(2^n*factorial(n)*w0y))
    eic = exp(im*k*z)
    function f(x, y)
        x, y = (x-xc)*cosθ0 - (y-yc)*sinθ0, (y-yc)*cosθ0 + (x-xc)*sinθ0
        x2 = x^2
        y2 = y^2
        v = sqrt((q0x*q0y)/(qzx*qzy))*exp(im*k*(x2/(2*qzx) + y2/(2*qzy)))
        v *= C*Hm(sqrt(2)*x/wzx)*Hn(sqrt(2)*y/wzy)*eiψG
        if constant_phase
            v *= eic
        end
        v
    end
    f
end

function gaussian(w0x, w0y, λ, z; xc = 0.0, yc = 0.0, θ0 = 0.0, constant_phase = true)
    hermite_gaussian(
        w0x,
        w0y,
        λ,
        z,
        0,
        0;
        xc = xc,
        yc = yc,
        θ0 = θ0,
        constant_phase = constant_phase
    )
end

function gaussian(w0, λ, z; xc = 0.0, yc = 0.0, θ0 = 0.0, constant_phase = true)
    gaussian(w0, w0, λ, z; xc = xc, yc = yc, θ0 = θ0, constant_phase = constant_phase)
end

function hermite_gaussian(
        w0,
        λ,
        z,
        m::Integer,
        n::Integer;
        xc = 0.0,
        yc = 0.0,
        θ0 = 0.0,
        constant_phase = true
)
    hermite_gaussian(
        w0,
        w0,
        λ,
        z,
        m,
        n;
        xc = xc,
        yc = yc,
        θ0 = θ0,
        constant_phase = constant_phase
    )
end

function hermite_gaussian(w0x, w0y, m::Integer, n::Integer; xc = 0.0, yc = 0.0, θ0 = 0.0)
    Hm = hermite_polynomial(m)
    Hn = hermite_polynomial(n)
    cosθ0 = cos(θ0)
    sinθ0 = sin(θ0)
    C = sqrt(sqrt(2/π)/(2^m*factorial(m)*w0x)*sqrt(2/π)/(2^n*factorial(n)*w0y))
    function f(x, y)
        x, y = (x-xc)*cosθ0 - (y-yc)*sinθ0, (y-yc)*cosθ0 + (x-xc)*sinθ0
        x2 = x^2
        y2 = y^2
        C*Hm(sqrt(2)*x/w0x)*Hn(sqrt(2)*y/w0y)*exp(-(x2/w0x^2 + y2/w0y^2))
    end
    f
end

function gaussian(w0x, w0y; xc = 0.0, yc = 0.0, θ0 = 0.0)
    hermite_gaussian(w0x, w0y, 0, 0; xc = xc, yc = yc, θ0 = θ0)
end

function gaussian(w0; xc = 0.0, yc = 0.0)
    gaussian(w0, w0; xc = xc, yc = yc)
end

function hermite_gaussian(w0, m::Integer, n::Integer; xc = 0.0, yc = 0.0, θ0 = 0.0)
    hermite_gaussian(w0, w0, m, n; xc = xc, yc = yc, θ0 = θ0)
end

function triangle_positions(np::Integer, px, py; θ0 = 0.0, xc = 0.0, yc = 0.0)
    xypvec = Tuple{Float64, Float64}[]
    cosθ = cos(θ0)
    sinθ = sin(θ0)
    for i in np:-1:1
        for j in 1:i
            xp = ((i-1)-(np-1)/2)*px
            yp = ((j-1)-(np-1)/2)*py
            push!(xypvec, (xp*cosθ-yp*sinθ+xc, xp*sinθ+yp*cosθ+yc))
        end
    end
    xypvec
end

function gaussian_modes(
        ::Type{T},
        w0,
        nx,
        ny,
        dx,
        dy,
        xycvec::AbstractVector{<:Tuple{<:Number, <:Number}};
        normalize = true
) where {T <: Number}
    n_modes = length(xycvec)
    modes = zeros(T, (nx, ny, n_modes))
    xvec, yvec = spatial_vectors(nx, ny, dx, dy)
    for (k, (xc, yc)) in enumerate(xycvec)
        mode = gaussian(w0; xc = xc, yc = yc).(xvec, yvec')
        if normalize
            mode ./= norm(mode)
        end
        modes[:, :, k] .= mode
    end
    modes
end

function gaussian_modes(
        w0,
        nx,
        ny,
        dx,
        dy,
        xycvec::AbstractVector{<:Tuple{<:Number, <:Number}};
        normalize = true
)
    gaussian_modes(ComplexF64, w0, nx, ny, dx, dy, xycvec; normalize = normalize)
end

function hermite_gaussian_modes(
        ::Type{T},
        w0,
        nx,
        ny,
        dx,
        dy,
        n_groups::Integer;
        θ0 = 0.0,
        normalize = true
) where {T <: Number}
    @assert n_groups >= 0
    n_modes = div((n_groups*(n_groups+1)), 2)
    modes = zeros(T, (nx, ny, n_modes))
    xvec, yvec = spatial_vectors(nx, ny, dx, dy)
    k = 0
    for m in 0:(n_groups - 1)
        for n in 0:(n_groups - m - 1)
            mode = hermite_gaussian(w0, m, n; θ0 = θ0).(xvec, yvec')
            if normalize
                mode ./= norm(mode)
            end
            k = k+1
            modes[:, :, k] .= mode
        end
    end
    modes
end

function hermite_gaussian_modes(
        w0,
        nx,
        ny,
        dx,
        dy,
        n_groups::Integer;
        θ0 = 0.0,
        normalize = true
)
    hermite_gaussian_modes(
        ComplexF64,
        w0,
        nx,
        ny,
        dx,
        dt,
        n_groups;
        θ0 = θ0,
        normalize = normalize
    )
end
