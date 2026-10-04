struct ScalarWaveField{U, T} <: AbstractField{U, 2}
    E::U
    dzE::U
    ds::NTuple{2, Float64}
    lambda::Float64
end

Functors.@functor ScalarWaveField (E, dzE)

function +(u::ScalarWaveField, v::ScalarWaveField)
    ScalarWaveField(u.E + v.E, u.dzE + v.dzE, u.ds, u.lambda)
end

function +(a::NamedTuple{(:E, :dzE, :ds, :lambda)}, b::ScalarWaveField)
    E = isnothing(a.E) ? b.E : a.E + b.E
    dzE = isnothing(a.dzE) ? b.dzE : a.dzE + b.dzE
    ScalarWaveField(E, dzE, b.ds, b.lambda)
end

+(b::ScalarWaveField, a::NamedTuple{(:E, :dzE, :ds, :lambda)}) = a + b

function ScalarModeBasis(fx::Real, fy::Real, λ::T, n0::Number) where {T <: Real}
    k0 = 2π / λ
    kx = 2π * fx
    ky = 2π * fy
    ScalarModeBasis(Complex{T}(sqrt(complex((k0 * n0)^2 - kx^2 - ky^2))))
end

function ScalarMediumModes(u::ScalarWaveField, medium)
    ScalarMediumModes(u.E, u.ds, u.lambda, medium)
end

function ScalarWaveField(u::U, ds::NTuple{2, Real}, λ::Real,
                         medium::Union{Number, ScalarMediumModes} = 1.0;
                         direction::Direction = Forward()
                         ) where {N, T, U <: AbstractArray{Complex{T}, N}}
    @assert N >= 2
    modes = ScalarMediumModes(u, ds, λ, medium).modes
    E_f = fft(u, (1, 2))
    dzE_f = similar(E_f)
    StructArray((dzE_f,)) .= apply_admittance.(modes, direction, E_f)
    ScalarWaveField(E_f, dzE_f, ds, λ)
end

function split_field(u::ScalarWaveField, medium::Union{Number, ScalarMediumModes} = 1.0)
    split_field(u, ScalarMediumModes(u, medium).modes)
end

function Base.ndims(u::ScalarWaveField, spatial::Bool = false)
    spatial ? 2 : ndims(u.electric)
end

Base.size(u::ScalarWaveField) = size(u.electric)

Base.size(u::ScalarWaveField, k::Integer) = size(u.electric, k)

Base.eltype(u::ScalarWaveField) = eltype(u.electric)

function set_field_data(u::ScalarWaveField, E::AbstractArray, dzE::AbstractArray)
    ScalarWaveField(E, dzE, u.ds, u.lambda)
end

flux_weight(u::ScalarWaveField) = prod(u.ds) / prod(size(u)[1:2]) * u.lambda / 2π

function power(u::ScalarWaveField, medium::Union{Number, ScalarMediumModes} = 1.0)
    power(u, ScalarMediumModes(u, medium).modes)
end
