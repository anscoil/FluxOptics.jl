struct HelmholtzField{U, L} <: AbstractField{U, 2}
    electric:: U
    electric_dz:: U
    ds:: NTuple{2, Float64}
    lambdas:: L
end

Functors.@functor HelmholtzField (electric, electric_dz)

function HelmholtzField(electric::U,
                        electric_dz::U,
                        ds::NTuple{2, Real},
                        lambdas::Union{Real, AbstractArray{<:Real}}
                        ) where {N, T, U <: AbstractArray{Complex{T}, N}}
    @assert N >= 2
    lambdas = parse_lambdas(electric, lambdas, 2)
    HelmholtzField(electric, electric_dz, ds, lambdas)
end

function HelmholtzField(nd::NTuple{N, Integer},
                        ds::NTuple{2, Real},
                        lambdas::Union{Real, AbstractArray{<:Real}}) where {N}
    electric = zeros(ComplexF64, nd)
    electric_dz = zeros(ComplexF64, nd)
    HelmholtzField(electric, electric_dz, ds, lambdas)
end

function compute_kz(fx::Real, fy::Real, λ::T, n0::Real = 1.0) where {T <: Real}
    k0 = 2π / λ
    kx = 2π * fx
    ky = 2π * fy
    Complex{T}(sqrt(Complex((k0 * n0)^2 - kx^2 - ky^2)))
end

function compute_kz(u::HelmholtzField{U}, n0::Real = 1.0) where {U}
    ns = size(u.electric)[1:2]
    K = similar(U, real, 1)
    fx = fftfreq(ns[1], 1/u.ds[1]) |> K
    fy = fftfreq(ns[2], 1/u.ds[2]) |> K
    compute_kz.(fx, fy', u.lambdas.val, n0)
end

function HelmholtzField(u::U,
                        ds::NTuple{2, Real},
                        lambdas::Union{Real, AbstractArray{<:Real}};
                        n0::Real = 1.0,
                        forward::Bool = true
                        ) where {N, T, U <: AbstractArray{Complex{T}, N}}
    @assert N >= 2
    lambdas = parse_lambdas(u, lambdas, 2)
    ns = size(u)[1:2]
    K = similar(U, real, 1)
    fx = fftfreq(ns[1], 1/ds[1]) |> K
    fy = fftfreq(ns[2], 1/ds[2]) |> K
    kz = compute_kz.(fx, fy', lambdas.val, n0)
    E_f = fft(u, (1, 2))
    sgn = forward ? 1 : -1
    @. E_f *= sgn * im * kz
    dEdz = ifft!(E_f, (1, 2))
    HelmholtzField(u, dEdz, ds, lambdas)
end

function HelmholtzField(u::ScalarField; n0::Real = 1.0, forward::Bool = true)
    HelmholtzField(u.electric, Tuple(u.ds), u.lambdas.collection; n0, forward)
end

function forward_field(u::HelmholtzField; n0::Real = 1.0)
    kz = compute_kz(u, n0)
    E_f = fft(u.electric, (1, 2))
    dEdz_f = fft(u.electric_dz, (1, 2))
    Eplus = ifft((@. (E_f + dEdz_f / (im * kz)) / 2), (1, 2))
    ScalarField(Eplus, u.ds, u.lambdas.collection)
end

function backward_field(u::HelmholtzField; n0::Real = 1.0)
    kz = compute_kz(u, n0)
    E_f = fft(u.electric, (1, 2))
    dEdz_f = fft(u.electric_dz, (1, 2))
    Eminus = ifft((@. (E_f - dEdz_f / (im * kz)) / 2), (1, 2))
    ScalarField(Eminus, u.ds, u.lambdas.collection)
end

function split_field(u::HelmholtzField; n0::Real = 1.0)
    u_fwd = forward_field(u; n0)
    u_bwd = set_field_data(u_fwd, u.electric .- u_fwd.electric)
    (u_fwd, u_bwd)
end

get_lambdas(u::HelmholtzField) = u.lambdas.val

get_lambdas_collection(u::HelmholtzField) = u.lambdas.collection

function Base.ndims(u::HelmholtzField, spatial::Bool = false)
    spatial ? 2 : ndims(u.electric)
end

Base.size(u::HelmholtzField) = size(u.electric)

Base.size(u::HelmholtzField, k::Integer) = size(u.electric, k)

Base.eltype(u::HelmholtzField) = eltype(u.electric)

function Base.copy(u::HelmholtzField)
    HelmholtzField(copy(u.electric), copy(u.electric_dz), u.ds, deepcopy(u.lambdas))
end

function Base.similar(u::HelmholtzField)
    HelmholtzField(similar(u.electric), similar(u.electric_dz), u.ds, deepcopy(u.lambdas))
end

function set_field_data(u::HelmholtzField,
                        electric::AbstractArray, electric_dz::AbstractArray)
    HelmholtzField(electric, electric_dz, u.ds, u.lambdas.collection)
end

function poynting_flux(u::HelmholtzField)
    T = real(eltype(u.electric))
    ds = T(prod(u.ds))
    imag.(sum(conj.(u.electric) .* u.electric_dz; dims = (1, 2))) .* ds
end

function power(u::HelmholtzField; n0::Real = 1.0)
    nx, ny = size(u)[1:2]
    kz = compute_kz(u, n0)
    E_f = fft(u.electric, (1, 2))
    dEdz_f = fft(u.electric_dz, (1, 2))
    Eplus_f = @. (E_f + dEdz_f / (im * kz)) / 2
    Eminus_f = @. (E_f - dEdz_f / (im * kz)) / 2
    T = real(eltype(u.electric))
    nrm = T(prod(u.ds) / (nx * ny))
    Pplus = sum(real.(kz) .* abs2.(Eplus_f);  dims = (1, 2)) .* nrm
    Pminus = sum(real.(kz) .* abs2.(Eminus_f); dims = (1, 2)) .* nrm
    (Pplus, Pminus)
end

function normalize_power!(u::HelmholtzField, target_power = 1;
                          n0::Real = 1.0, forward::Bool = true)
    Pplus, Pminus = power(u; n0)
    P = forward ? Pplus : Pminus
    scale = sqrt.(target_power ./ P)
    u.electric .*= scale
    u.electric_dz .*= scale
    u
end
