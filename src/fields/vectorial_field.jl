struct VectorialField{U} <: AbstractField{U, 2}
    Ex::U
    Ey::U
    Hx::U
    Hy::U
    ds:: NTuple{2, Float64}
    lambda::Float64
end

Functors.@functor VectorialField (Ex, Ey, Hx, Hy)

struct ZDecEpsilon{T}
    xx::T; xy::T
    yx::T; yy::T
    zz::T
end

struct Epsilon{T}
    xx::T; xy::T; xz::T
    yx::T; yy::T; yz::T
    zx::T; zy::T; zz::T
end

Base.broadcastable(ϵ::Union{ZDecEpsilon, Epsilon}) = Ref(ϵ)

function ZDecEpsilon(ϵ::Epsilon)
    ZDecEpsilon(ϵ.xx, ϵ.xy, ϵ.yx, ϵ.yy, ϵ.zz)
end

function compute_P(kx::Real, ky::Real, k0::T, ϵ::Number) where {T <: Real}
    p11 = kx * ky / (k0 * ϵ)
    p12 = k0 - kx^2 / (k0 * ϵ)
    p21 = -k0 + ky^2 / (k0 * ϵ)
    p22 = -kx * ky / (k0 * ϵ)
    SMatrix{2, 2, Complex{T}}((p11, p21, p12, p22))
end

function compute_P(kx::Real, ky::Real, k0::T,
                   ϵ::Union{ZDecEpsilon, Epsilon}) where {T <: Real}
    compute_P(kx, ky, k0, ϵ.zz)
end

function compute_Q0(kx::Real, ky::Real, k0::T) where {T <: Real}
    q11 = -kx * ky / k0
    q12 = kx^2 / k0
    q21 = -ky^2 / k0
    q22 = kx * ky / k0
    SMatrix{2, 2, Complex{T}}((q11, q21, q12, q22))
end

function compute_Q(kx::Real, ky::Real, k0::T, ϵ::Number) where {T <: Real}
    Q = compute_Q0(kx, ky, k0)
    Q + SMatrix{2, 2, Complex{T}}((0, k0 * ϵ, -k0 * ϵ, 0))
end

function compute_Q(kx::Real, ky::Real, k0::T, ϵ::ZDecEpsilon) where {T <: Real}
    Q = compute_Q0(kx, ky, k0)
    q11 = -k0 * ϵ.yx
    q12 = -k0 * ϵ.yy
    q21 = k0 * ϵ.xx
    q22 = k0 * ϵ.xy
    Q + SMatrix{2, 2, Complex{T}}((q11, q21, q12, q22))
end

function compute_Q(kx::Real, ky::Real, k0::T, ϵ::Epsilon) where {T <: Real}
    Q = compute_Q(kx, ky, k0, ZDecEpsilon(ϵ))
    q11 = k0 * (ϵ.yz * ϵ.zx) / ϵ.zz
    q12 = k0 * (ϵ.yz * ϵ.zy) / ϵ.zz
    q21 = -k0 * (ϵ.xz * ϵ.zx) / ϵ.zz
    q22 = -k0 * (ϵ.xz * ϵ.zy) / ϵ.zz
    Q + SMatrix{2, 2, Complex{T}}((q11, q21, q12, q22))
end

function compute_D0(kx::Real, ky::Real, k0::T) where {T <: Real}
    SMatrix{2, 2, Complex{T}}((0, 0, 0, 0))
end

function compute_D1(kx::Real, ky::Real, k0::T,
                    ϵ::Union{Number, ZDecEpsilon}) where {T <: Real}
    compute_D0(kx, ky, k0)
end

function compute_D1(kx::Real, ky::Real, k0::T, ϵ::Epsilon) where {T <: Real}
    d11 = -kx * ϵ.zx / ϵ.zz
    d12 = -kx * ϵ.zy / ϵ.zz
    d21 = -ky * ϵ.zx / ϵ.zz
    d22 = -ky * ϵ.zy / ϵ.zz
    SMatrix{2, 2, Complex{T}}((d11, d21, d12, d22))
end

function compute_D2(kx::Real, ky::Real, k0::T,
                    ϵ::Union{Number, ZDecEpsilon}) where {T <: Real}
    compute_D0(kx, ky, k0)
end

function compute_D2(kx::Real, ky::Real, k0::T, ϵ::Epsilon) where {T <: Real}
    d11 = -ky * ϵ.yz / ϵ.zz
    d12 = kx * ϵ.yz / ϵ.zz
    d21 = ky * ϵ.xz / ϵ.zz
    d22 = -kx * ϵ.xz / ϵ.zz
    SMatrix{2, 2, Complex{T}}((d11, d21, d12, d22))
end

function compute_M(kx::Real, ky::Real, k0::T, ϵ) where {T <: Real}
    P = compute_P(kx, ky, k0, ϵ)
    Q = compute_Q(kx, ky, k0, ϵ)
    D1 = compute_D1(kx, ky, k0, ϵ)
    D2 = compute_D2(kx, ky, k0, ϵ)
    [[D1 P]; [Q D2]]
end

function eigen_modes(fx::Real, fy::Real, λ::T, ϵ) where {T <: Real}
    k0 = 2π / λ
    kx = 2π * fx
    ky = 2π * fy
    M = Matrix(compute_M(kx, ky, k0, ϵ))
    tol = sqrt(eps(Float64)) * norm(M)
    q(v) = abs(imag(v)) <= tol ? 0 : (imag(v) > 0 ? -1 : 1)
    F = eigen(M; sortby = λ -> (q(λ), -real(λ)))
    V = SMatrix{4, 4, Complex{T}}(F.vectors)
    SVector{4, Complex{T}}(F.values), V, inv(V)
end

function eigen_modes(u::U, ds::NTuple{2, Real}, λ::Real, ϵ
                     ) where {N, T, U <: AbstractArray{Complex{T}, N}}
    @assert N >= 2
    ns = size(u)[1:2]
    fx = fftfreq(ns[1], 1/ds[1])
    fy = fftfreq(ns[2], 1/ds[2])
    M = eigen_modes.(fx, fy', T(λ), ϵ)
    adapt(get_backend(u),
          (; kz = getindex.(M, 1), P = getindex.(M, 2), P_inv = getindex.(M, 3)))
end

function eigen_modes(u::VectorialField{U}, λ::Real, ϵ) where {U}
    eigen_modes(u.Ex, u.ds, λ, ϵ)
end
