module Fields

using Functors
using AbstractFFTs
using LinearAlgebra
using StaticArrays
using StructArrays
using KernelAbstractions
using Adapt
using ..FluxOptics
using ..FluxOptics: isbroadcastable, bzip

import Base: +, -, *, /

export AbstractField, ScalarField, ScalarWaveField, VectorialField
export ScalarWaveState, VectorialState
export get_lambdas, get_lambdas_collection
export get_tilts, get_tilts_collection, offset_tilts!
export select_lambdas, select_tilts, set_field_ds!, set_field_data, set_field_tilts
export is_on_axis
export power, normalize_power!, coupling_efficiency, intensity, phase
export orthonormalize, unitary_transform, spatial_moments, spatial_centroids, spatial_variance
export AbstractModeBasis, ModeBasis, ScalarModeBasis
export MediumModes, VectorialMediumModes, ScalarMediumModes
export Permittivity, ZDecEpsilon, Epsilon, FresnelCoefficients, permittivity
export eigenvalues, mode_indices, compute_fresnel, transmission, reflection
export split_field, electric_field, poynting_flux, normalize_poynting!
export decompose, recompose, decompose_adjoint, recompose_adjoint, project, project_adjoint
export Direction, Forward, Backward, isforward, isbackward

function parse_val(u::AbstractArray{Complex{T}, N},
                   val::AbstractArray,
                   Nd::Integer) where {N, T}
    shape = ntuple(k -> k <= Nd ? 1 : size(val, k - Nd), N)
    val_adapt = similar(u, T, shape)
    copyto!(val_adapt, val)
    @assert isbroadcastable(val_adapt, u)
    val_adapt
end

function parse_lambdas(u::U, lambdas, Nd::Integer) where {T, U <: AbstractArray{Complex{T}}}
    lambdas_collection = isa(lambdas, Real) ? T(lambdas) : T.(lambdas)
    lambdas_val = isa(lambdas, Real) ? T(lambdas) : parse_val(u, lambdas, Nd)
    (; val = lambdas_val, collection = lambdas_collection)
end

function parse_tilts(u::U, tilts, Nd::Integer) where {T, U <: AbstractArray{Complex{T}}}
    tilts_collection = map(θ -> isa(θ, Real) ? T.([θ]) : T.(θ), tilts)
    tilts_val = map(θ -> parse_val(u, isa(θ, Real) ? [θ] : θ, Nd), tilts)
    (; val = tilts_val, collection = tilts_collection)
end

abstract type AbstractField{U, Nd} end

abstract type Direction end

struct Forward <: Direction end

struct Backward <: Direction end

Base.broadcastable(d::Direction) = Ref(d)

Base.reverse(::Type{Forward}) = Backward
Base.reverse(::Type{Backward}) = Forward
Base.reverse(::Forward) = Backward()
Base.reverse(::Backward) = Forward()
Base.reverse(l, ::Forward) = l
Base.reverse(l, ::Backward) = reverse(l)

Base.sign(::Type{Forward}) = 1
Base.sign(::Type{Backward}) = -1
Base.sign(::Forward) = 1
Base.sign(::Backward) = -1

isforward(::Forward) = true
isforward(::Backward) = false
isbackward(::Forward) = false
isbackward(::Backward) = true

Base.similar(u::AbstractField) = fmap(similar, u)
Base.zero(u::AbstractField) = fmap(zero, u)
Base.copy(u::AbstractField) = fmap(copy, u)

function Base.copyto!(u::AbstractField, v::AbstractField)
    fmap(copyto!, u, v)
    u
end

abstract type AbstractModeBasis end

struct ModeBasis{K, M} <: AbstractModeBasis
    kz::K
    P::M
    P_inv::M
end

eigenvalues(m::ModeBasis) = m.kz
basis(m::ModeBasis) = m.P
basis_inv(m::ModeBasis) = m.P_inv

mul_im(z::Complex) = Complex(-imag(z), real(z))

struct ScalarModeBasis{T} <: AbstractModeBasis
    kz::T
    inv_kz::T
end

eigenvalues(m::ScalarModeBasis) = SVector(m.kz, -m.kz)

function basis(m::ScalarModeBasis)
    o, ikz = one(m.kz), mul_im(m.kz)
    @SMatrix [o o; ikz -ikz]
end

function basis_inv(m::ScalarModeBasis)
    o, u = one(m.kz), -mul_im(m.inv_kz)
    @SMatrix([o u; o -u]) / 2
end

electric_indices(::ModeBasis) = SVector(1, 2)  # E⊥
magnetic_indices(::ModeBasis) = SVector(3, 4)  # H⊥

electric_indices(::ScalarModeBasis) = SVector(1)  # E
magnetic_indices(::ScalarModeBasis) = SVector(2)  # ∂zE

mode_indices(m::AbstractModeBasis, ::Forward) = electric_indices(m)
mode_indices(m::AbstractModeBasis, ::Backward) = magnetic_indices(m)

function decompose(m::AbstractModeBasis, Ψ::SVector, direction::Direction)
    basis_inv(m)[mode_indices(m, direction), :] * Ψ
end

function decompose(m::ScalarModeBasis, Ψ::SVector{2}, direction::Direction)
    SVector((Ψ[1] - sign(direction) * mul_im(m.inv_kz * Ψ[2])) / 2)
end

function recompose(m::AbstractModeBasis, a::SVector, direction::Direction)
    basis(m)[:, mode_indices(m, direction)] * a
end

function recompose(m::ScalarModeBasis, a::SVector{1}, direction::Direction)
    SVector(a[1], sign(direction) * mul_im(m.kz * a[1]))
end

function decompose_adjoint(m::AbstractModeBasis, ∂a::SVector, direction::Direction)
    basis_inv(m)[mode_indices(m, direction), :]' * ∂a
end

function decompose_adjoint(m::ScalarModeBasis, ∂a::SVector{1}, direction::Direction)
    SVector(∂a[1], sign(direction) * mul_im(conj(m.inv_kz) * ∂a[1])) / 2
end

function recompose_adjoint(m::AbstractModeBasis, ∂Ψ::SVector, direction::Direction)
    basis(m)[:, mode_indices(m, direction)]' * ∂Ψ
end

function recompose_adjoint(m::ScalarModeBasis, ∂Ψ::SVector{2}, direction::Direction)
    SVector(∂Ψ[1] - sign(direction) * mul_im(conj(m.kz) * ∂Ψ[2]))
end

function project(m::AbstractModeBasis, Ψ::SVector, direction::Direction)
    recompose(m, decompose(m, Ψ, direction), direction)
end

function project_adjoint(m::AbstractModeBasis, ∂Ψ::SVector, direction::Direction)
    decompose_adjoint(m, recompose_adjoint(m, ∂Ψ, direction), direction)
end

function split_state(m::AbstractModeBasis, ψ::Vararg{Number, N}) where {N}
    Ψ = SVector(ψ)
    Ψ_fwd = project(m, Ψ, Forward())
    Tuple(vcat(Ψ_fwd, Ψ - Ψ_fwd))
end

function split_field(u::AbstractField, modes::AbstractArray{<:AbstractModeBasis})
    data, rebuild = Functors.functor(u)
    fwd, bwd = map(similar, data), map(similar, data)
    StructArray((Tuple(fwd)..., Tuple(bwd)...)) .= split_state.(modes, Tuple(data)...)
    rebuild(fwd), rebuild(bwd)
end

function admittance(m::AbstractModeBasis, direction::Direction)
    c = mode_indices(m, direction)
    P = basis(m)
    P[magnetic_indices(m), c] * inv(P[electric_indices(m), c])
end

function apply_admittance(m::AbstractModeBasis, direction::Direction,
                          e::Vararg{Number, N}) where {N}
    Tuple(admittance(m, direction) * SVector(e))
end

struct MediumModes{B <: AbstractModeBasis, S}
    modes::S
end

MediumModes{B}(modes::S) where {B, S} = MediumModes{B, S}(modes)

function MediumModes{B}(u::AbstractArray, ds::NTuple{2, Real}, λ::Real, medium) where {B}
    @assert ndims(u) >= 2
    T = real(eltype(u))
    fx = fftfreq(size(u, 1), 1 / ds[1])
    fy = fftfreq(size(u, 2), 1 / ds[2])
    modes = StructArray(B(x, y, T(λ), medium) for x in fx, y in fy)
    MediumModes{B}(adapt(get_backend(u), modes))
end

function MediumModes{B}(u::AbstractArray, ds::NTuple{2, Real}, λ::Real,
                        medium::MediumModes{B}) where {B}
    medium
end

const VectorialMediumModes = MediumModes{ModeBasis}
const ScalarMediumModes = MediumModes{ScalarModeBasis}

poynting_density(e, dze) = imag(conj(e) * dze)
poynting_density(ex, ey, hx, hy) = real(ex * conj(hy) - ey * conj(hx))

function flux_weight end

function poynting_flux(u::AbstractField)
    T = real(eltype(u))
    data = Tuple(Functors.children(u))
    sum(poynting_density.(data...); dims = (1, 2)) .* T(flux_weight(u))
end

function directional_fluxes(m::AbstractModeBasis, ψ::Vararg{Number, N}) where {N}
    Ψ = SVector(ψ)
    Ψ_fwd = project(m, Ψ, Forward())
    (poynting_density(Tuple(Ψ_fwd)...), -poynting_density(Tuple(Ψ - Ψ_fwd)...))
end

function power(u::AbstractField, modes::AbstractArray{<:AbstractModeBasis})
    T = real(eltype(u))
    data = Tuple(Functors.children(u))
    P_fwd, P_bwd = similar(first(data), T), similar(first(data), T)
    StructArray((P_fwd, P_bwd)) .= directional_fluxes.(modes, data...)
    c = T(flux_weight(u))
    (sum(P_fwd; dims = (1, 2)) .* c, sum(P_bwd; dims = (1, 2)) .* c)
end

function rescale!(u::AbstractField, s)
    foreach(a -> a .*= s, Functors.children(u))
    u
end

function normalize_power!(u::AbstractField, medium::Union{Number, ScalarMediumModes} = 1.0,
                          target_power = 1; direction::Direction = Forward())
    P_fwd, P_bwd = power(u, medium)
    rescale!(u, sqrt.(target_power ./ (isforward(direction) ? P_fwd : P_bwd)))
end

function normalize_poynting!(u::AbstractField, S_out = 1)
    S_in = poynting_flux(u)
    rescale!(u, @. sqrt(abs(S_out / S_in)))
end

struct FresnelCoefficients{R}
    r12::R
    t21::R
    t12::R
    r21::R
end

function FresnelCoefficients(P1::SMatrix{N, N}, P2::SMatrix{N, N}) where {N}
    lo = SVector(ntuple(identity, Val(N ÷ 2)))
    hi = SVector(ntuple(i -> i + N ÷ 2, Val(N ÷ 2)))
    S = hcat(-P1[:, hi], P2[:, lo]) \ hcat(P1[:, lo], -P2[:, hi])
    FresnelCoefficients(S[lo, lo], S[lo, hi], S[hi, lo], S[hi, hi])
end

function FresnelCoefficients(m1::AbstractModeBasis, m2::AbstractModeBasis)
    FresnelCoefficients(basis(m1), basis(m2))
end

transmission(f::FresnelCoefficients, ::Forward) = f.t12
transmission(f::FresnelCoefficients, ::Backward) = f.t21
reflection(f::FresnelCoefficients, ::Forward) = f.r21
reflection(f::FresnelCoefficients, ::Backward) = f.r12

function compute_fresnel(m1::MediumModes{B}, m2::MediumModes{B}) where {B}
    FresnelCoefficients.(m1.modes, m2.modes)
end

include("scalar_field.jl")

include("scalar_wave_field.jl")

include("vectorial_field.jl")

electric_field(u::ScalarWaveField) = ifft(u.E, (1, 2))
electric_field(u::VectorialField) = (ifft(u.Ex, (1, 2)), ifft(u.Ey, (1, 2)))

function electric_field(u::AbstractField, medium, direction::Direction)
    u_fwd, u_bwd = split_field(u, medium)
    electric_field(isforward(direction) ? u_fwd : u_bwd)
end

end
