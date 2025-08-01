function rs_kernel(
        normalization_factor::T, x::T, y::T, λ::T, z::T) where {T <: AbstractFloat}
    k = T(2π)/λ
    r = sqrt(x*x + y*y + z*z)
    normalization_factor*(exp(im*k*r)/r)*(z/r)*(1/r-im*k)
end

struct RSKernel{T, K, V} <: AbstractFourierKernel{T, K}
    s_vec::V
    kernel_cache::Union{Nothing, LRU{T, K}}
    normalization_factor::T
    z::T

    function RSKernel(
            U::Type{<:AbstractArray{Complex{T}, N}},
            nx::Integer,
            ny::Integer,
            dx::Real,
            dy::Real,
            z::Real,
            lambdas::Union{Nothing, Tuple{Vararg{<:Real}}} = nothing
    ) where {T <: Real, N}
        @assert N >= 2
        F = adapt_dim(U, 1, real)
        Nx, Ny = 2*nx-1, 2*ny-1
        x = circshift((1 - nx):(nx - 1), nx) .* dx |> F
        y = circshift((1 - ny):(ny - 1), ny) .* dy |> F
        s_vec = (; x = x, y = y)
        V = typeof(s_vec)
        normalization_factor = T(dx*dx/2π)
        if isnothing(lambdas)
            new{T, Nothing, V}(s_vec, nothing, normalization_factor, z)
        else
            K = adapt_dim(U, 2)
            kernel_cache = LRU{T, K}(maxsize = length(lambdas))
            for λ in lambdas
                kernel = fft!(@. rs_kernel(normalization_factor, x, y', T(λ), T(z)))
                kernel_cache[T(λ)] = kernel
            end
            new{T, K, V}(s_vec, kernel_cache, normalization_factor, z)
        end
    end

    function RSKernel(
            u::U,
            dx::Real,
            dy::Real,
            z::Real,
            lambdas::Union{Nothing, Tuple{Vararg{<:Real}}} = nothing
    ) where {U <: AbstractArray{<:Complex}}
        nx, ny = size(u)
        RSKernel(typeof(u), nx, ny, dx, dy, z, lambdas)
    end
end

function apply_kernel!(u::AbstractArray, rs_k::RSKernel{T, Nothing}, lambdas,
        direction::Type{<:Direction}) where {T}
    x, y = rs_k.s_vec.x, rs_k.s_vec.y
    nrm_f = rs_k.normalization_factor
    kernel = fft!(@. rs_kernel(nrm_f, x, y', lambdas, rs_k.z))
    u .*= kernel_direction.(kernel, direction)
end

function apply_kernel!(u::AbstractArray, rs_k::RSKernel{T, K}, λ::Real,
        direction::Type{<:Direction}) where {T, K <: AbstractArray}
    kernel_key = T(λ)
    if haskey(rs_k.kernel_cache, kernel_key)
        @. u *= kernel_direction(rs_k.kernel_cache[kernel_key], direction)
    else
        x, y = rs_k.s_vec.x, rs_k.s_vec.y
        nrm_f = rs_k.normalization_factor
        kernel = fft!(@. rs_kernel(nrm_f, x, y', T(λ), rs_k.z))
        rs_k.kernel_cache[kernel_key] = kernel
        @. u *= kernel_direction(kernel, direction)
    end
    u
end

struct RSProp{M, K, U, P} <: AbstractPropagator{M, K}
    kernel::K
    p_f::P
    u_tmp::U

    function RSProp(u::U,
            dx::Real,
            dy::Real,
            z::Real,
            lambdas::Union{Nothing, Tuple{Vararg{<:Real}}} = nothing
    ) where {N, U <: AbstractArray{<:Complex, N}}
        @assert N >= 2
        nx, ny = size(u)
        kernel = RSKernel(u, dx, dy, z, lambdas)
        u_tmp = similar(u, (2*nx-1, 2*ny-1, size(u)[3:end]...))
        p_f = make_fft_plans(u_tmp, (1, 2))
        new{Static, typeof(kernel), U, typeof(p_f)}(kernel, p_f, u_tmp)
    end

    function RSProp(u::AbstractArray{<:Complex, N},
            dx::Real,
            dy::Real,
            z::Real,
            lambda::Real) where {N}
        @assert N >= 2
        RSProp(u, dx, dy, z, (lambda,))
    end

    function RSProp(u::ScalarField,
            dx::Real,
            dy::Real,
            z::Real,
            kernel_cache::Bool = false)
        lambdas = kernel_cache ? Tuple(unique(u.lambdas)) : nothing
        RSProp(u.data, dx, dy, z, lambdas)
    end
end

function _propagate_core!(apply_kernel_fn, u::AbstractArray, p::RSProp)
    nx, ny = size(u)
    p.u_tmp .= 0
    u_view = @view p.u_tmp[1:nx, 1:ny, ..]
    copyto!(u_view, u)
    p.p_f.ft * p.u_tmp
    apply_kernel_fn(p.u_tmp)
    p.p_f.ift * p.u_tmp
    copyto!(u, u_view)
    u
end
