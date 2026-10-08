struct VectorialField{U} <: AbstractField{U, 2}
    Ex::U
    Ey::U
    Hx::U
    Hy::U
    ds:: NTuple{2, Float64}
    lambda::Float64
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

const Permittivity = Union{Number, ZDecEpsilon, Epsilon}

function Epsilon(A::AbstractMatrix)
    size(A) == (3, 3) || throw(ArgumentError("permittivity tensor must be 3×3"))
    Epsilon(A[1, 1], A[1, 2], A[1, 3],
            A[2, 1], A[2, 2], A[2, 3],
            A[3, 1], A[3, 2], A[3, 3])
end

is_z_decoupled(A::AbstractMatrix) = all(iszero, (A[1, 3], A[2, 3], A[3, 1], A[3, 2]))

function ZDecEpsilon(A::AbstractMatrix)
    size(A) == (3, 3) || throw(ArgumentError("permittivity tensor must be 3×3"))
    is_z_decoupled(A) ||
        throw(ArgumentError("tensor is not z-decoupled: xz, yz, zx, zy must be zero"))
    ZDecEpsilon(A[1, 1], A[1, 2], A[2, 1], A[2, 2], A[3, 3])
end

permittivity(n::Number) = n^2

function permittivity(n::NTuple{3, Number}, R::AbstractMatrix = I)
    permittivity(R * Diagonal(SVector(n) .^ 2) * transpose(R))
end

permittivity(A::AbstractMatrix) = is_z_decoupled(A) ? ZDecEpsilon(A) : Epsilon(A)

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

function ModeBasis(fx::Real, fy::Real, λ::T, ϵ::Permittivity) where {T <: Real}
    k0 = 2π / λ
    kx = 2π * fx
    ky = 2π * fy
    M = Matrix(compute_M(kx, ky, k0, ϵ))
    tol = sqrt(eps(Float64)) * norm(M)
    q(μ) = abs(imag(μ)) <= tol ? 0 : (imag(μ) > 0 ? -1 : 1)
    F = eigen(M; sortby = μ -> (q(μ), -real(μ)))
    V = SMatrix{4, 4, Complex{T}}(F.vectors)
    ModeBasis(SVector{4, Complex{T}}(F.values), V, inv(V))
end

function VectorialMediumModes(u::VectorialField, medium)
    VectorialMediumModes(u.Ex, u.ds, u.lambda, medium)
end

function VectorialField(Ex::U, Ey::U, ds::NTuple{2, Real}, λ::Real,
                        medium::Union{Permittivity, VectorialMediumModes} = 1.0;
                        direction::Direction = Forward()
                        ) where {N, T, U <: AbstractArray{Complex{T}, N}}
    @assert N >= 2 && size(Ex) == size(Ey)
    modes = VectorialMediumModes(Ex, ds, λ, medium).modes
    Ex_f = fft(Ex, (1, 2))
    Ey_f = fft(Ey, (1, 2))
    Hx_f, Hy_f = similar(Ex_f), similar(Ey_f)
    StructArray((Hx_f, Hy_f)) .= apply_admittance.(modes, direction, Ex_f, Ey_f)
    VectorialField(Ex_f, Ey_f, Hx_f, Hy_f, ds, λ)
end

function VectorialField(E::U, jones::NTuple{2, Number}, ds::NTuple{2, Real}, λ::Real,
                        medium::Union{Permittivity, VectorialMediumModes} = 1.0;
                        direction::Direction = Forward()
                        ) where {T, U <: AbstractArray{Complex{T}}}
    jx, jy = Complex{T}.(jones)
    VectorialField(jx .* E, jy .* E, ds, λ, medium; direction)
end

function split_field(u::VectorialField,
                     medium::Union{Permittivity, VectorialMediumModes} = 1.0)
    split_field(u, VectorialMediumModes(u, medium).modes)
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

flux_weight(u::VectorialField) = prod(u.ds) / prod(size(u)[1:2])

function power(u::VectorialField, medium::Union{Permittivity, VectorialMediumModes} = 1.0)
    power(u, VectorialMediumModes(u, medium).modes)
end

struct VectorialState{C} <: FieldVector{4, C}
    Ex::C
    Ey::C
    Hx::C
    Hy::C
end

function StructArrays.StructArray(u::VectorialField)
    StructArray{VectorialState{eltype(u.Ex)}}((u.Ex, u.Ey, u.Hx, u.Hy))
end
