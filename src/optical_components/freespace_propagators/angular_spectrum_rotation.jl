function field_rotation_matrix(θx::Real, θy::Real; initial_tilts = (0, 0))
    # Based on Rodrigues' rotation formula
    θx0, θy0 = initial_tilts
    if isapprox(θx, θx0) && isapprox(θy, θy0)
        return I(3)
    end
    k0 = [sin(θx0), sin(θy0), sqrt(1 - (sin(θx0)^2 + sin(θy0)^2))]
    kf = [sin(θx), sin(θy), sqrt(1 - (sin(θx)^2 + sin(θy)^2))]
    kr = cross(k0, kf)
    Ksinθ = [0 -kr[3] kr[2]; kr[3] 0 -kr[1]; -kr[2] kr[1] 0]
    K = Ksinθ ./ norm(kr)
    cosθ = dot(k0, kf)
    I(3) .+ Ksinθ .+ (1-cosθ)*(K*K)
end

function prepare_vectors(u::AbstractArray{Complex{T}, 2}, dx::Real, dy::Real, λ::Real,
        k0v, kfv, M0, Mf) where {T <: Real}
    nx, ny = size(u)
    k0 = 2π/λ
    kx0, ky0 = k0v
    kxf, kyf = kfv
    M = Mf*inv(M0)
    xv, yv = spatial_vectors(nx, ny, dx, dy)
    x = similar(u, T)
    y = similar(u, T)
    copyto!(x, reshape(repeat(xv, ny), (nx, ny)))
    copyto!(y, repeat(yv', nx))
    kx = similar(u, T)
    ky = similar(u, T)
    copyto!(kx, 2π*reshape(repeat(fftfreq(nx, 1/dx), ny), (nx, ny)) .+ kx0)
    copyto!(ky, 2π*repeat(fftfreq(ny, 1/dy)', nx) .+ ky0)
    kz = @. sqrt(max(0, k0^2 - (kx^2 + ky^2)))
    kx′ = similar(u, T)
    ky′ = similar(u, T)
    @. kx′ = (M[1, 1]*kx + M[1, 2]*ky + M[1, 3]*kz)*dx
    @. ky′ = (M[2, 1]*kx + M[2, 2]*ky + M[2, 3]*kz)*dy
    kxc = (maximum(kx′)+minimum(kx′))/2
    kyc = (maximum(ky′)+minimum(ky′))/2
    kx′ .-= kxc
    ky′ .-= kyc
    kx .-= kx0
    ky .-= ky0
    kx .*= dx
    ky .*= dy
    detJ = sqrt(abs(Mf[3, 3]/M0[3, 3]))
    # detJ = sqrt(abs(det(Mf[1:2, 1:2])/det(M0[1:2, 1:2])))
    (x, y), (kx, ky), (kx′, ky′), (kxc-kxf*dx, kyc-kyf*dy), detJ
end

function prepare_vectors(u::AbstractArray{Complex{T}, 2}, dx::Real, dy::Real, λ::Real,
        Mf, direction::Type{<:Direction} = Forward; M0 = I(3)) where {T <: Real}
    M0, Mf == direction == Forward ? (M0, Mf) : (Mf, M0)
    k0 = 2π/λ
    k0v = [0, 0, k0]
    kx0, ky0, _ = M0*k0v
    kxf, kyf, _ = Mf*k0v
    prepare_vectors(u, dx, dy, λ, (kx0, ky0), (kxf, kyf), M0, Mf)
end

function prepare_vectors(u::AbstractArray{Complex{T}, 2}, dx::Real, dy::Real, λ::Real,
        θs::Tuple{Real, Real}, direction::Type{<:Direction} = Forward;
        initial_tilts = (0, 0)) where {T <: Real}
    k0 = 2π/λ
    (θx0, θy0), (θx, θy) = direction == Forward ? (initial_tilts, θs) : (θs, initial_tilts)
    kx0, ky0 = k0*sin(θx0), k0*sin(θy0)
    kxf, kyf = k0*sin(θx), k0*sin(θy)
    M0 = field_rotation_matrix(θx0, θy0)
    Mf = field_rotation_matrix(θx, θy)
    prepare_vectors(u, dx, dy, λ, (kx0, ky0), (kxf, kyf), M0, Mf)
end

function as_correct_shift(u::AbstractArray{Complex{T}, 2}, kxy, kxy′, kxyc,
        direction::Type{<:Direction}) where {T <: Real}
    nx, ny = size(u)
    kx, ky = kxy
    kx′, ky′ = kxy′
    if iseven(nx)
        @. u *= conj_direction(cis(T(0.5)*(kx′-kx)), direction)
    end
    if iseven(ny)
        @. u *= conj_direction(cis(T(0.5)*(ky′-ky)), direction)
    end
end

function as_nufft2d2!(u::AbstractArray{Complex{T}, 2}, kxy, kxy′,
        ::Type{Forward}, eps) where {T <: Real}
    nufft2d2!(kxy..., reshape(u, (:, 1)), -1, eps, u)
end

function as_nufft2d2!(u::AbstractArray{Complex{T}, 2}, kxy, kxy′,
        ::Type{Backward}, eps) where {T <: Real}
    nufft2d2!(kxy′..., reshape(u, (:, 1)), -1, eps, u)
end

function as_nufft2d1!(u::AbstractArray{Complex{T}, 2}, kxy, kxy′,
        ::Type{Forward}, eps) where {T <: Real}
    nufft2d1!(kxy′..., u, 1, eps, u)
end

function as_nufft2d1!(u::AbstractArray{Complex{T}, 2}, kxy, kxy′,
        ::Type{Backward}, eps) where {T <: Real}
    nufft2d1!(kxy..., u, 1, eps, u)
end

function as_tilt_compensation(xy, λ::Real, Mf::AbstractMatrix{<:Real},
        direction::Type{<:Direction}; M0 = I(3))
    M0, Mf = direction == Forward ? (M0, Mf) : (Mf, M0)
    k0 = 2π/λ
    k0v = [0, 0, k0]
    kx0, ky0, _ = M0*k0v
    kxf, kyf, _ = Mf*k0v
    x, y = xy
    pretilt = conj_direction((@. cis(-(kx0 .* x .+ ky0 .* y))), direction)
    posttilt = conj_direction((@. cis(kxf .* x .+ kyf .* y)), direction)
    direction == Forward ? (pretilt, posttilt) : (posttilt, pretilt)
end

function as_tilt_compensation(xy, λ::Real, θs::Tuple{Real, Real},
        direction::Type{<:Direction}; initial_tilts = (0, 0))
    θ0, θf = direction == Forward ? (initial_tilts, θs) : (θs, initial_tilts)
    k0 = 2π/λ
    kx0, ky0 = @. k0*sin(θ0)
    kxf, kyf = @. k0*sin(θf)
    x, y = xy
    pretilt = conj_direction((@. cis(-(kx0 .* x .+ ky0 .* y))), direction)
    posttilt = conj_direction((@. cis(kxf .* x .+ kyf .* y)), direction)
    direction == Forward ? (pretilt, posttilt) : (posttilt, pretilt)
end

function as_nufft_rotation!(u, kxy, kxy′, kxyc, s, direction, eps)
    nx, ny = size(u)
    kxc, kyc = kxyc
    nxr = (0:(nx - 1)) .- (nx-1)/2
    nyr = (0:(ny - 1)) .- (ny-1)/2
    if direction == Backward
        @. u *= cis(-kxc*nxr - kyc*nyr')
    end
    as_nufft2d2!(u, kxy, kxy′, direction, eps)
    as_correct_shift(u, kxy, kxy′, kxyc, direction)
    as_nufft2d1!(u, kxy, kxy′, direction, eps)
    if direction == Forward
        @. u *= cis(kxc*nxr + kyc*nyr')
    end
    u .*= s/length(u)
    # nufft2d3!(x./T(dx), y./T(dy), u, -1, eps, kx, ky, reshape(u, (:, 1)))
    # nufft2d3!(kx′, ky′, u, 1, eps, x./T(dx), y./T(dy), reshape(u, (:, 1)))
end

function as_rotation!(u, kxy, kxy′, kxyc, tilts, s, direction, eps, compensate_tilt)
    pretilt, posttilt = tilts
    if !compensate_tilt
        u .*= pretilt
    end
    as_nufft_rotation!(u, kxy, kxy′, kxyc, s, direction, eps)
    if !compensate_tilt
        u .*= posttilt
    end
    u
end

function as_rotation!(
        u::AbstractArray{Complex{T}, 2}, ds::Tuple{Real, Real}, λ::Real,
        Mf, direction::Type{<:Direction} = Forward;
        M0 = I(3), eps = 2e-7, compensate_tilt::Bool = true) where {T <: Real}
    xy, kxy, kxy′, kxyc, s = prepare_vectors(u, ds..., λ, Mf, direction; M0)
    tilts = as_tilt_compensation(xy, λ, Mf, direction; M0)
    as_rotation!(u, kxy, kxy′, kxyc, tilts, s, direction, eps, compensate_tilt)
end

function as_rotation!(
        u::AbstractArray{Complex{T}, 2}, ds::Tuple{Real, Real}, λ::Real,
        θs::Tuple{Real, Real}, direction::Type{<:Direction} = Forward;
        initial_tilts = (0, 0), eps = 2e-7, compensate_tilt::Bool = true) where {T <: Real}
    xy, kxy, kxy′, kxyc, s = prepare_vectors(u, ds..., λ, θs, direction; initial_tilts)
    tilts = as_tilt_compensation(xy, λ, θs, direction; initial_tilts)
    as_rotation!(u, kxy, kxy′, kxyc, tilts, s, direction, eps, compensate_tilt)
end

function as_rotation(u::U, ds::Tuple{Real, Real}, λ::Real, Mf::AbstractMatrix{<:Real},
        direction::Type{<:Direction} = Forward; M0 = I(3), eps = 2e-7,
        compensate_tilt::Bool = true
) where {T <: Real, U <: AbstractArray{Complex{T}, 2}}
    as_rotation!(copy(u), ds, λ, Mf, direction; eps, compensate_tilt, M0)
end

function as_rotation(u::U, ds::Tuple{Real, Real}, λ::Real, θs::Tuple{Real, Real},
        direction::Type{<:Direction} = Forward; initial_tilts = (0, 0), eps = 2e-7,
        compensate_tilt::Bool = true
) where {T <: Real, U <: AbstractArray{Complex{T}, 2}}
    as_rotation!(copy(u), ds, λ, θs, direction; eps, compensate_tilt, initial_tilts)
end

function make_nufft_plan(u::AbstractArray{Complex{T}, 2}, ns::Tuple{Integer, Integer},
        s::Tuple{AbstractMatrix, AbstractMatrix}, type::Integer,
        isign::Integer, eps::Real) where {T <: Real}
    p_nft = finufft_makeplan(type, [ns...], isign, 1, eps; dtype = T)
    finufft_setpts!(p_nft, s...)
    p_nft
end

function exec_nufft_plan!(p, u::AbstractArray{Complex{T}, 2}) where {T <: Real}
    nx, ny = size(u)
    u_in = reshape(u, (nx, ny, 1))
    u_out = reshape(u, (:, 1))
    if p.type == 2
        finufft_exec!(p, u_in, u_out)
    end
    if p.type == 1
        finufft_exec!(p, u_out, u_in)
    end
    if p.type == 3
        finufft_exec!(p, u_out, u_out)
    end
end

# function plan_as_rotation(
#         u::AbstractArray{Complex{T}, 2}, ds::Tuple{Real, Real}, λ::Real,
#         M::AbstractArray{<:Real, 2}; eps = 2e-7) where {T <: Real}
#     nx, ny = size(u)
#     (x, y), kxy, kxy′ = prepare_vectors(u, ds..., λ, M)
#     p_nft = make_nufft_plan(u, (nx, ny), kxy, 2, -1, eps)
#     p_nift = make_nufft_plan(u, (nx, ny), kxy′, 1, 1, eps)
#     k0 = 2π/λ
#     kx0, ky0 = M*[0, 0, k0]
#     ((p_nft, p_nift), kxy, kxy′,
#         (@. Complex{T}(cis(x .* kx0 .+ y .* ky0))),
#         sqrt(abs(M[3, 3])))
# end

# function as_rotation!(u::AbstractArray{Complex{T}, 2}, p;
#         compensate_tilt::Bool = true) where {T <: Real}
#     nx, ny = size(u)
#     (p_nft, p_nift), (kx, ky), (kx′, ky′), tilt_correction, s = p
#     exec_nufft_plan!(p_nft, u)
#     if iseven(nx)
#         @. u *= cis(T(0.5)*(kx′-kx))
#     end
#     if iseven(ny)
#         @. u *= cis(T(0.5)*(ky′-ky))
#     end
#     exec_nufft_plan!(p_nift, u)
#     if !compensate_tilt
#         u .*= tilt_correction
#     end
#     u .*= s/(nx*ny)
# end

# function as_rotation(u::AbstractArray{Complex{T}, 2}, p;
#         compensate_tilt::Bool = true) where {T <: Real}
#     as_rotation!(copy(u), p; compensate_tilt)
# end

function rotate_scalar_field!(u::ScalarField{U, 2}, Mf::AbstractMatrix{<:Real},
        direction::Type{<:Direction}; eps = 2e-7, compensate_tilt = true
) where {U <: AbstractArray{<:Complex, 2}}
    @assert isreal(u.lambdas.val) && all(isreal.(u.tilts.val))
    θs = Tuple(asin.(Mf[1:2, 3]))
    M0 = field_rotation_matrix(u.tilts.val...)
    as_rotation!(u.electric, u.ds, u.lambdas.val, Mf, direction;
        eps, compensate_tilt, M0)
    ScalarField(u.electric, u.ds, u.lambdas.val; tilts = θs)
end

function rotate_scalar_field!(u::ScalarField{U, 2}, θs::Tuple{Real, Real},
        direction::Type{<:Direction}; eps = 2e-7, compensate_tilt = true
) where {U <: AbstractArray{<:Complex, 2}}
    @assert isreal(u.lambdas.val) && all(isreal.(u.tilts.val))
    as_rotation!(u.electric, u.ds, u.lambdas.val, θs, direction;
        eps, compensate_tilt, initial_tilts = u.tilts.val)
    ScalarField(u.electric, u.ds, u.lambdas.val; tilts = θs)
end

array_from_view(v) = v

# Annoying hack because of FINUFFT
function array_from_view(v::SubArray{T}) where {T}
    @assert Base.iscontiguous(v)
    ptr = Base.unsafe_convert(Ptr{T}, pointer(v))
    len = length(v)
    sz = size(v)
    return Base.unsafe_wrap(Array, ptr, sz; own = false)
end

function as_rotation!(u::ScalarField{U, 2}, Mf::AbstractMatrix{<:Real},
        direction::Type{<:Direction} = Forward; eps = 2e-7, compensate_tilt = true
) where {U <: AbstractArray{<:Complex}}
    θs = Tuple(asin.(Mf[1:2, 3]))
    foreach(
        v -> rotate_scalar_field!(
            set_field_data(v, array_from_view(v.electric)),
            Mf, direction; eps, compensate_tilt),
        vec(u))
    ScalarField(u.electric, u.ds, u.lambdas.collection; tilts = θs)
end

function as_rotation!(u::ScalarField{U, 2}, θs::Tuple{Real, Real},
        direction::Type{<:Direction} = Forward; eps = 2e-7, compensate_tilt = true
) where {U <: AbstractArray{<:Complex}}
    foreach(
        v -> rotate_scalar_field!(
            set_field_data(v, array_from_view(v.electric)),
            θs, direction; eps, compensate_tilt),
        vec(u))
    ScalarField(u.electric, u.ds, u.lambdas.collection; tilts = θs)
end

function as_rotation(u::ScalarField{U, 2}, Mf::AbstractMatrix{<:Real},
        direction::Type{<:Direction} = Forward; eps = 2e-7, compensate_tilt = true
) where {U <: AbstractArray{<:Complex}}
    as_rotation!(copy(u), Mf, direction; eps, compensate_tilt)
end

function as_rotation(u::ScalarField{U, 2}, θs::Tuple{Real, Real},
        direction::Type{<:Direction} = Forward; eps = 2e-7, compensate_tilt = true
) where {U <: AbstractArray{<:Complex}}
    as_rotation!(copy(u), θs, direction; eps, compensate_tilt)
end
