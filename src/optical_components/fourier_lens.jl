function fourier_lens_kernel(
        fx::T, fy::T, λ::T, θx::T, θy::T, nrm_f::T) where {T <: AbstractFloat}
    exp(-im*(fx^2*θx + fy^2*θy)/λ)*nrm_f/λ
end

function fourier_lens_chirp(
        x::T, y::T, λ::T, θx::T, θy::T) where {T <: AbstractFloat}
    exp(im*(x^2*θx + y^2*θy)/λ)
end

struct FourierLens{M, K, T, U, V, P} <: AbstractPropagator{M, K}
    θx::T
    θy::T
    nrm_f::T
    s_vec::V
    f_vec::V
    p_f::P
    u_tmp::U

    # Warning: aliasing expected if nx*dx*dx′/(λ*fl) > 2 || ny*dy*dy′/(λ*fl) > 2
    # but this should not be a relevant use case.
    function FourierLens(u::U,
            dx::Real,
            dy::Real,
            dx′::Real,
            dy′::Real,
            fl::Real
    ) where {N, T, U <: AbstractArray{Complex{T}, N}}
        @assert N >= 2
        @assert fl > 0
        nx, ny = size(u)
        Nx, Ny = 2*nx-1, 2*ny-1
        θx = T(-π*(dx′/dx)/fl)
        θy = T(-π*(dy′/dy)/fl)
        nrm_f = T(dx*dy/fl)
        F = adapt_dim(U, 1, real)
        x = ((0:(Nx - 1)) .- (nx-1)/2) .* dx |> F
        y = ((0:(Ny - 1)) .- (ny-1)/2) .* dy |> F
        s_vec = (; x = x, y = y)
        fx = circshift((1 - nx):(nx - 1), nx) .* dx |> F
        fy = circshift((1 - ny):(ny - 1), ny) .* dy |> F
        f_vec = (; x = fx, y = fy)
        V = typeof(f_vec)
        u_tmp = similar(u, (2*nx-1, 2*ny-1, size(u)[3:end]...))
        p_f = make_fft_plans(u_tmp, (1, 2))
        P = typeof(p_f)
        new{Static, Nothing, T, U, V, P}(θx, θy, nrm_f, s_vec, f_vec, p_f, u_tmp)
    end

    function FourierLens(u::U,
            dx::Real,
            dy::Real,
            dx′::Real,
            dy′::Real,
            fl::Real,
            λ::Real
    ) where {N, T, U <: AbstractArray{Complex{T}, N}}
        @assert N >= 2
        @assert fl > 0
        nx, ny = size(u)
        θx = T(-2π*dx*dx′/(λ*fl))
        θy = T(-2π*dy*dy′/(λ*fl))
        nrm_f = T(dx*dy/(λ*fl))
        a = (1, 1)
        w = ((1, θx), (1, θy))
        p_czt = plan_czt!(u, (1, 2), a, w; center_on_grid = true)
        p_f = (; czt = p_czt, iczt = adjoint(p_czt))
        P = typeof(p_f)
        new{Static, Val{T(λ)}, T, Nothing, Nothing, P}(
            θx, θy, nrm_f, nothing, nothing, p_f, nothing)
    end

    function FourierLens(u::ScalarField, dx::Real, dy::Real, dx′::Real,
            dy′::Real, fl::Real, cache::Bool = false)
        if cache
            λ = only(unique(u.lambdas)) # may fail, but that's what we want
            FourierLens(u.data, dx, dy, dx′, dy′, fl, λ)
        else
            FourierLens(u.data, dx, dy, dx′, dy′, fl)
        end
    end
end

function apply_chirp!(u::AbstractArray, p::FourierLens{M, Nothing}, lambdas,
        direction::Type{<:Direction}) where {M}
    x, y = p.s_vec.x, p.s_vec.y
    @. u *= kernel_direction(fourier_lens_chirp(x, y', lambdas, p.θx, p.θy), direction)
end

function apply_kernel!(u::AbstractArray, p::FourierLens{M, Nothing}, lambdas,
        direction::Type{<:Direction}) where {M}
    fx, fy = p.f_vec.x, p.f_vec.y
    kernel = fft!(@. fourier_lens_kernel(fx, fy', lambdas, p.θx, p.θy, p.nrm_f))
    u .*= kernel_direction.(kernel, direction)
end

function propagate!(
        u::ScalarField, p::FourierLens{M, Nothing}, direction::Type{<:Direction}) where {M}
    p.u_tmp .= 0
    u_view = view(p.u_tmp, axes(u.data)...)
    copyto!(u_view, u.data)
    apply_chirp!(p.u_tmp, p, u.lambdas_collection, direction)
    p.p_f.ft * p.u_tmp
    apply_kernel!(p.u_tmp, p, u.lambdas_collection, direction)
    p.p_f.ift * p.u_tmp
    apply_chirp!(p.u_tmp, p, u.lambdas_collection, direction)
    copyto!(u.data, u_view)
    u
end

function get_czt_direction(p::FourierLens{M, Val{λ}}, direction::Type{Forward}) where {M, λ}
    p.p_f.czt
end

function get_czt_direction(
        p::FourierLens{M, Val{λ}}, direction::Type{Backward}) where {M, λ}
    p.p_f.iczt
end

function propagate!(u::AbstractArray, p::FourierLens{M, Val{λ}},
        direction::Type{<:Direction}) where {M, λ}
    get_czt_direction(p, direction) * u
    u .*= p.nrm_f
end

function propagate!(u::ScalarField, p::FourierLens{M, Val{λ}},
        direction::Type{<:Direction}) where {M, λ}
    λu = only(unique(u.lambdas))
    @assert λu == λ
    propagate!(u.data, p, direction)
    u
end
