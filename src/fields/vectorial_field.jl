struct VectorialField{U, T} <: AbstractField{U, 2}
    Ex::U
    Ey::U
    Hx::U
    Hy::U
    ds:: NTuple{2, T}
    lambda::T
end

Functors.@functor VectorialField (Ex, Ey, Hx, Hy)

function +(u::VectorialField, v::VectorialField)
    VectorialField(u.Ex + v.Ex, u.Ey + v.Ey, u.Hx + v.Hx, u.Hy + v.Hy, u.ds, u.lambda)
end

function +(a::NamedTuple{(:Ex, :Ey, :Hx, :Hy, :ds, :lambda)}, b::VectorialField)
    Ex = isnothing(a.Ex) ? b.Ex : a.Ex + b.Ex
    Ey = isnothing(a.Ey) ? b.Ey : a.Ey + b.Ey
    Hx = isnothing(a.Hx) ? b.Hx : a.Hx + b.Hx
    Hy = isnothing(a.Hy) ? b.Hy : a.Hy + b.Hy
    VectorialField(Ex, Ey, Hx, Hy, b.ds, b.lambda)
end

+(b::VectorialField, a::NamedTuple{(:Ex, :Ey, :Hx, :Hy, :ds, :lambda)}) = a + b

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

Permittivity = Union{Number, ZDecEpsilon, Epsilon}

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
    (; kz = SVector{4, Complex{T}}(F.values), P = V, P_inv = inv(V))
end

function eigen_modes(u::U, ds::NTuple{2, Real}, λ::Real, ϵ
                     ) where {N, T, U <: AbstractArray{Complex{T}, N}}
    @assert N >= 2
    ns = size(u)[1:2]
    fx = fftfreq(ns[1], 1/ds[1])
    fy = fftfreq(ns[2], 1/ds[2])
    modes = StructArray(eigen_modes(x, y, T(λ), ϵ) for x in fx, y in fy)
    adapt(get_backend(u), modes)
end

function eigen_modes(u::VectorialField{U}, ϵ) where {U}
    eigen_modes(u.Ex, u.ds, u.lambda, ϵ)
end

function fresnel_modal(P1::SMatrix{4, 4}, P2::SMatrix{4, 4})
    fw, bw = SVector(1, 2), SVector(3, 4)
    S = hcat(-P1[:, bw], P2[:, fw]) \ hcat(P1[:, fw], -P2[:, bw])
    (; r12 = S[fw, fw], t21 = S[fw, bw], t12 = S[bw, fw], r21 = S[bw, bw])
end

fresnel_modal(m1::NamedTuple, m2::NamedTuple) = fresnel_modal(m1.P, m2.P)

compute_fresnel(modes_1::StructArray, modes_2::StructArray) = fresnel_modal.(modes_1, modes_2)

mode_indices(::Forward) = SVector(1, 2)
mode_indices(::Backward) = SVector(3, 4)

function decompose(m, Ψ::SVector{4}, direction::Direction)
    m.P_inv[mode_indices(direction), :] * Ψ
end

function recompose(m, a::SVector{2}, direction::Direction)
    m.P[:, mode_indices(direction)] * a
end

function decompose_adjoint(m, ∂a::SVector{2}, direction::Direction)
    m.P_inv[mode_indices(direction), :]' * ∂a
end

function recompose_adjoint(m, ∂Ψ::SVector{4}, direction::Direction)
    m.P[:, mode_indices(direction)]' * ∂Ψ
end

function project(m, Ψ::SVector{4}, direction::Direction)
    recompose(m, decompose(m, Ψ, direction), direction)
end

function admittance(m, direction::Direction)
    c = mode_indices(direction)
    m.P[SVector(3, 4), c] * inv(m.P[SVector(1, 2), c])
end

function apply_admittance(m, direction::Direction, ex::Number, ey::Number)
    Tuple(admittance(m, direction) * SVector(ex, ey))
end

function VectorialField(Ex::U, Ey::U, ds::NTuple{2, Real}, λ::Real, ϵ = 1.0;
                        direction::Direction = Forward(),
                        modes = eigen_modes(Ex, ds, λ, ϵ)
                        ) where {N, T, U <: AbstractArray{Complex{T}, N}}
    @assert N >= 2 && size(Ex) == size(Ey)
    Ex_f = fft(Ex, (1, 2))
    Ey_f = fft(Ey, (1, 2))
    Hx_f, Hy_f = similar(Ex_f), similar(Ey_f)
    StructArray((Hx_f, Hy_f)) .= apply_admittance.(modes, direction, Ex_f, Ey_f)
    VectorialField(Ex_f, Ey_f, Hx_f, Hy_f, T.(ds), T(λ))
end

function split_state(m, ex, ey, hx, hy)
    Ψ = SVector(ex, ey, hx, hy)
    Ψ_fwd = project(m, Ψ, Forward())
    Tuple(vcat(Ψ_fwd, Ψ - Ψ_fwd))
end

function split_field(u::VectorialField, ϵ = 1.0;
                     modes = eigen_modes(u.Ex, u.ds, u.lambda, ϵ))
    fwd = map(similar, (u.Ex, u.Ey, u.Hx, u.Hy))
    bwd = map(similar, (u.Ex, u.Ey, u.Hx, u.Hy))
    StructArray((fwd..., bwd...)) .= split_state.(modes, u.Ex, u.Ey, u.Hx, u.Hy)
    VectorialField(fwd..., u.ds, u.lambda), VectorialField(bwd..., u.ds, u.lambda)
end

function Base.ndims(u::VectorialField, spatial::Bool = false)
    spatial ? 2 : ndims(u.Ex)
end

Base.size(u::VectorialField) = size(u.Ex)

Base.size(u::VectorialField, k::Integer) = size(u.Ex, k)

Base.eltype(u::VectorialField) = eltype(u.Ex)

function set_field_data(u::VectorialField, Ex, Ey, Hx, Hy)
    VectorialField(Ex, Ey, Hx, Hy, u.ds, u.lambda)
end

poynting_density(ex, ey, hx, hy) = real(ex * conj(hy) - ey * conj(hx))
poynting_density(Ψ::SVector{4}) = poynting_density(Ψ...)

function directional_fluxes(m, ex, ey, hx, hy)
    Ψ = SVector(ex, ey, hx, hy)
    Ψ_fwd = project(m, Ψ, Forward())
    (poynting_density(Ψ_fwd), -poynting_density(Ψ - Ψ_fwd))
end

function poynting_flux(u::VectorialField)
    T = real(eltype(u))
    c = T(prod(u.ds)) / prod(size(u)[1:2])
    sum(poynting_density.(u.Ex, u.Ey, u.Hx, u.Hy); dims = (1, 2)) .* c
end

function power(u::VectorialField, ϵ = 1.0;
               modes = eigen_modes(u.Ex, u.ds, u.lambda, ϵ))
    T = real(eltype(u))
    c = T(prod(u.ds)) / prod(size(u)[1:2])
    P_fwd, P_bwd = similar(u.Ex, T), similar(u.Ex, T)
    StructArray((P_fwd, P_bwd)) .= directional_fluxes.(modes, u.Ex, u.Ey, u.Hx, u.Hy)
    (sum(P_fwd; dims = (1, 2)) .* c, sum(P_bwd; dims = (1, 2)) .* c)
end
