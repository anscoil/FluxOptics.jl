struct ScalarWaveField{U, L} <: AbstractField{U, 2}
    electric::U
    electric_dz::U
    ds::NTuple{2, Float64}
    lambdas::L
end

Functors.@functor ScalarWaveField (electric, electric_dz)

function Base.:+(a::NamedTuple{(:electric, :electric_dz, :ds, :lambdas)}, b::ScalarWaveField)
    electric = isnothing(a.electric) ? b.electric : a.electric + b.electric
    electric_dz = isnothing(a.electric_dz) ? b.electric_dz : a.electric_dz + b.electric_dz
    ScalarWaveField(electric, electric_dz, b.ds, b.lambdas)
end

Base.:+(b::ScalarWaveField, a::NamedTuple{(:electric, :electric_dz, :ds, :lambdas)}) = a + b

function compute_kz(fx::Real, fy::Real, λ::T, n0::Number = 1.0) where {T <: Real}
    k0 = 2π / λ
    kx = 2π * fx
    ky = 2π * fy
    Complex{T}(sqrt(complex((k0 * n0)^2 - kx^2 - ky^2)))
end

function compute_kz(u::U, ds::NTuple{2, Real}, lambdas, n0::Number = 1.0
                    ) where {N, T, U <: AbstractArray{Complex{T}, N}}
    @assert N >= 2
    ns = size(u)[1:2]
    K = similar(U, real, 1)
    fx = fftfreq(ns[1], 1/ds[1]) |> K
    fy = fftfreq(ns[2], 1/ds[2]) |> K
    compute_kz.(fx, fy', lambdas, n0)
end

function compute_kz(u::ScalarWaveField, n0::Number = 1.0)
    compute_kz(u.electric, u.ds, u.lambdas.val, n0)
end

function compute_kz(u::ScalarField{U, 2}, n0::Number = 1.0) where {U}
    compute_kz(u.electric, Tuple(u.ds), u.lambdas.val, n0)
end

function compute_fresnel_r12(u::ScalarWaveField, n1::Number, n2::Number)
    kz1 = compute_kz(u.electric, u.ds, u.lambdas.val, n1)
    kz2 = compute_kz(u.electric, u.ds, u.lambdas.val, n2)
    r12 = @. (kz1 - kz2) / (kz1 + kz2)
end

function compute_fresnel_t12(u::ScalarWaveField, n1::Number, n2::Number)
    kz1 = compute_kz(u.electric, u.ds, u.lambdas.val, n1)
    kz2 = compute_kz(u.electric, u.ds, u.lambdas.val, n2)
    t12 = @. 2*kz1 / (kz1 + kz2)
end

function ScalarWaveField(u::U,
                         ds::NTuple{2, Real},
                         lambdas::Union{Real, AbstractArray{<:Real}};
                         n0::Number = 1.0,
                         forward::Bool = true
                         ) where {N, T, U <: AbstractArray{Complex{T}, N}}
    @assert N >= 2
    lambdas = parse_lambdas(u, lambdas, 2)
    kz = compute_kz(u, ds, lambdas.val, n0)
    E_f = fft(u, (1, 2))
    sgn = forward ? 1 : -1
    dEdz_f = @. E_f * sgn * im * kz
    ScalarWaveField(E_f, dEdz_f, ds, lambdas)
end

function ScalarWaveField(u::ScalarField{U, 2};
                         n0::Number = 1.0, forward::Bool = true) where {U}
    ScalarWaveField(u.electric, Tuple(u.ds), u.lambdas.collection; n0, forward)
end

function ScalarWaveField(u_fwd::ScalarField{U, 2},
                         u_bwd::ScalarField{U, 2};
                         n0::Number = 1.0) where {U}
    lambdas = u_fwd.lambdas
    kz = compute_kz(u_fwd, n0)
    electric = u_fwd.electric .+ u_bwd.electric
    fft!(electric, (1, 2))
    electric_dz = u_fwd.electric .- u_bwd.electric
    fft!(electric_dz, (1, 2))
    @. electric_dz *= im * kz
    ds = u_fwd.ds
    ScalarWaveField(electric, electric_dz, Tuple(ds), lambdas)
end

function split_field(u::ScalarWaveField; n0::Number = 1.0)
    kz = compute_kz(u, n0)
    dEdz_f = @. u.electric_dz / (im * kz)
    Eplus = @. (u.electric + dEdz_f) / 2
    Eminus = @. (u.electric - dEdz_f) / 2
    ifft!(Eplus, (1, 2))
    ifft!(Eminus, (1, 2))
    u_fwd = ScalarField(Eplus, u.ds, u.lambdas.collection)
    u_bwd = ScalarField(Eminus, u.ds, u.lambdas.collection)
    (u_fwd, u_bwd)
end

get_lambdas(u::ScalarWaveField) = u.lambdas.val

get_lambdas_collection(u::ScalarWaveField) = u.lambdas.collection

function Base.ndims(u::ScalarWaveField, spatial::Bool = false)
    spatial ? 2 : ndims(u.electric)
end

Base.size(u::ScalarWaveField) = size(u.electric)

Base.size(u::ScalarWaveField, k::Integer) = size(u.electric, k)

Base.eltype(u::ScalarWaveField) = eltype(u.electric)

function set_field_data(u::ScalarWaveField,
                        electric::AbstractArray, electric_dz::AbstractArray)
    ScalarWaveField(electric, electric_dz, u.ds, deepcopy(u.lambdas))
end

function poynting_flux(u::ScalarWaveField)
    T = real(eltype(u))
    c = T(prod(u.ds)) / prod(size(u)[1:2])
    k0 = T.(2π ./ u.lambdas.val)
    S = imag.(sum(conj.(u.electric) .* u.electric_dz; dims = (1, 2)))
    S .* c ./ k0
end

function power(u::ScalarWaveField; n0::Number = 1.0)
    T = real(eltype(u))
    c = T(prod(u.ds)) / prod(size(u)[1:2])
    k0 = T.(2π ./ u.lambdas.val)
    kz = compute_kz(u, n0)
    E, dE = u.electric, u.electric_dz
    P_fwd = sum(@. real(kz) * abs2((E + dE / (im * kz)) / 2); dims = (1, 2))
    P_bwd = sum(@. real(kz) * abs2((E - dE / (im * kz)) / 2); dims = (1, 2))
    (P_fwd .* c ./ k0, P_bwd .* c ./ k0)
end

function +(u::ScalarWaveField, v::ScalarWaveField)
    set_field_data(u, u.electric + v.electric, u.electric_dz + v.electric_dz)
end
