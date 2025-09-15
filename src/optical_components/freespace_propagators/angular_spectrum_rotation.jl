function prepare_vectors(u::AbstractArray{Complex{T}, 2}, λ::Real,
        M::AbstractArray{<:Real, 2}, compensate_tilt::Bool) where {T <: Real}
    nx, ny = size(u)
    @assert isapprox(det(M), 1)
    k0 = 2π/λ
    kx0, ky0 = compensate_tilt ? M*[0, 0, k0] : (0, 0)
    xv, yv = spatial_vectors(nx, ny, 1, 1)
    x = similar(u, T)
    y = similar(u, T)
    copyto!(x, reshape(repeat(xv, ny), (nx, ny)))
    copyto!(y, repeat(yv', nx))
    kx = similar(u, T)
    ky = similar(u, T)
    copyto!(kx, 2π*reshape(repeat(fftfreq(nx, 1), ny), (nx, ny)))
    copyto!(ky, 2π*repeat(fftfreq(ny, 1)', nx))
    kz = @. sqrt(max(0, k0^2 - (kx^2 + ky^2)))
    kx′ = similar(u, T)
    ky′ = similar(u, T)
    @. kx′ = (M[1, 1]*kx + M[1, 2]*ky + M[1, 3]*kz) - kx0
    @. ky′ = (M[2, 1]*kx + M[2, 2]*ky + M[2, 3]*kz) - ky0
    (x, y), (kx, ky), (kx′, ky′)
end

function as_rotation!(
        u::AbstractArray{Complex{T}, 2}, λ::Real, M::AbstractArray{<:Real, 2};
        eps = 2e-7, compensate_tilt::Bool = true, nthreads = 0) where {T <: Real}
    nx, ny = size(u)
    xy, kxy, kxy′ = prepare_vectors(u, λ, M, compensate_tilt)
    kx, ky = kxy
    kx′, ky′ = kxy′
    nufft2d2!(kxy..., reshape(u, (:, 1)), -1, eps, u)
    if iseven(nx)
        @. u *= cis(T(0.5)*(kx′-kx))
    end
    if iseven(ny)
        @. u *= cis(T(0.5)*(ky′-ky))
    end
    nufft2d1!(kxy′..., u, 1, eps, u)
    # nufft2d1 and nufft2d2 appear to be much faster even though they require a little
    # translation adjustment
    # nufft2d3!(xy..., u, -1, eps, kxy..., reshape(u, (:, 1)))
    # nufft2d3!(kxy′..., u, 1, eps, xy..., reshape(u, (:, 1)))
    u ./= nx*ny
end

function as_rotation(u::U, λ::Real, M::AbstractArray{<:Real, 2};
        eps = 2e-7, compensate_tilt::Bool = true,
        nthreads = 0
) where {T <: Real, U <: AbstractArray{Complex{T}, 2}}
    as_rotation!(copy(u), λ, M; eps, compensate_tilt, nthreads)
end

function plan_as_rotation(
        u::AbstractArray{Complex{T}, 2}, λ::Real, M::AbstractArray{<:Real, 2};
        eps = 2e-7, compensate_tilt::Bool = true, nthreads = 0) where {T <: Real}
    nx, ny = size(u)
    xy, kxy, kxy′ = prepare_vectors(u, λ, M, compensate_tilt)
    p_nft = finufft_makeplan(3, 2, -1, 1, eps; dtype = T, nthreads)
    finufft_setpts!(p_nft, xy..., T[], kxy...)
    p_nift = finufft_makeplan(3, 2, 1, 1, eps; dtype = T, nthreads)
    finufft_setpts!(p_nift, kxy′..., T[], xy...)
    p_nft, p_nift
end

function as_rotation!(p, u::AbstractArray{Complex{T}, 2}) where {T <: Real}
    p_nft, p_nift = p
    u_flat = reshape(u, (:, 1))
    finufft_exec!(p_nft, u_flat, u_flat)
    finufft_exec!(p_nift, u_flat, u_flat)
    u ./= length(u)
end

function as_rotation(p, u::AbstractArray{Complex{T}, 2}) where {T <: Real}
    as_rotation!(p, copy(u))
end
