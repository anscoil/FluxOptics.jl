using AbstractFFTs
using FFTW
using EllipsisNotation
using LRUCache

function make_fft_plans(
        u::U, dims::NTuple{N, Integer}) where {N, U <: AbstractArray{<:Complex}}
    p_ft = plan_fft!(u, dims, flags = FFTW.MEASURE)
    p_ift = plan_ifft!(u, dims, flags = FFTW.MEASURE)
    (; ft = p_ft, ift = p_ift)
end

function as_kernel(fx::T, fy::T, λ::T, z::T) where {T <: AbstractFloat}
    f² = complex(1/λ^2)
    exp(im*T(2)*π*z*sqrt(f² - fx*fx - (fy*fy)))
end

struct ASKernel{T, K, V} <: AbstractFourierKernel{T, K}
    f_vec::V
    kernel_cache::Union{Nothing, LRU{T, K}}
    z::T

    function ASKernel(
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
        fx = fftfreq(nx, T(1/dx)) |> F
        fy = fftfreq(ny, T(1/dy)) |> F
        f_vec = (; x = fx, y = fy)
        V = typeof(f_vec)
        if isnothing(lambdas)
            new{T, Nothing, V}(f_vec, nothing, z)
        else
            K = adapt_dim(U, 2)
            kernel_cache = LRU{T, K}(maxsize = length(lambdas))
            for λ in lambdas
                kernel_cache[T(λ)] = @. as_kernel(fx, fy', T(λ), T(z))
            end
            new{T, K, V}(f_vec, kernel_cache, z)
        end
    end

    function ASKernel(
            u::U,
            dx::Real,
            dy::Real,
            z::Real,
            lambdas::Union{Nothing, Tuple{Vararg{<:Real}}} = nothing
    ) where {U <: AbstractArray{<:Complex}}
        nx, ny = size(u)
        ASKernel(typeof(u), nx, ny, dx, dy, z, lambdas)
    end
end

function kernel_direction(kernel, ::Type{Forward})
    kernel
end

function kernel_direction(kernel, ::Type{Backward})
    conj(kernel)
end

function apply_kernel!(u::AbstractArray, as_k::ASKernel{T, Nothing}, lambdas,
        direction::Type{<:Direction}) where {T}
    fx, fy = as_k.f_vec.x, as_k.f_vec.y
    @. u *= kernel_direction(as_kernel(fx, fy', lambdas, as_k.z), direction)
end

function apply_kernel!(u::AbstractArray, as_k::ASKernel{T, K}, λ::Real,
        direction::Type{<:Direction}) where {T, K <: AbstractArray}
    kernel_key = T(λ)
    if haskey(as_k.kernel_cache, kernel_key)
        @. u *= kernel_direction(as_k.kernel_cache[kernel_key], direction)
    else
        fx, fy = as_k.f_vec.x, as_k.f_vec.y
        kernel = @. as_kernel(fx, fy', T(λ), as_k.z)
        as_k.kernel_cache[kernel_key] = kernel
        @. u *= kernel_direction(kernel, direction)
    end
    u
end

function apply_kernel!(u::AbstractArray,
        kernel::AbstractFourierKernel{T, K},
        lambdas::AbstractArray,
        direction::Type{<:Direction}) where {T, K <: AbstractArray}
    inds = CartesianIndices(size(u)[3:end])
    for i in inds
        apply_kernel!(@view(u[:, :, i]), kernel, lambdas[i], direction)
    end
    u
end

struct ASProp{M, K, P} <: AbstractPropagator{M, K}
    kernel::K
    p_f::P

    function ASProp(u::AbstractArray{<:Complex, N},
            dx::Real,
            dy::Real,
            z::Real,
            lambdas::Union{Nothing, Tuple{Vararg{<:Real}}} = nothing) where {N}
        @assert N >= 2
        kernel = ASKernel(u, dx, dy, z, lambdas)
        A_plan = similar(u)
        p_f = make_fft_plans(A_plan, (1, 2))
        new{Static, typeof(kernel), typeof(p_f)}(kernel, p_f)
    end

    function ASProp(u::AbstractArray{<:Complex, N},
            dx::Real,
            dy::Real,
            z::Real,
            lambda::Real) where {N}
        @assert N >= 2
        ASProp(u, dx, dy, z, (lambda,))
    end

    function ASProp(u::ScalarField,
            dx::Real,
            dy::Real,
            z::Real,
            kernel_cache::Bool = false)
        lambdas = kernel_cache ? Tuple(unique(u.lambdas)) : nothing
        ASProp(u.data, dx, dy, z, lambdas)
    end
end

function _propagate_core!(apply_kernel_fn, u::AbstractArray, p::ASProp)
    p.p_f.ft * u
    apply_kernel_fn(u)
    p.p_f.ift * u
end

function propagate!(u::AbstractArray, p::AbstractPropagator{M, K}, λ::Real,
        direction::Type{<:Direction}
) where {M, T, K <: AbstractFourierKernel{T}}
    _propagate_core!(u, p) do v
        apply_kernel!(v, p.kernel, T(λ), direction)
    end
    u
end

function propagate!(u::AbstractArray, p::AbstractPropagator{M, K},
        direction::Type{<:Direction}) where {
        M, T, K <: AbstractFourierKernel{T, <:AbstractArray}}
    kernel_cache = p.kernel.kernel_cache
    length(kernel_cache) == 1 ||
        error("Propagation kernel should hold exactly one wavelength")
    λ = first(keys(kernel_cache))
    propagate!(u, p, λ, direction)
end

function propagate!(u::AbstractArray, p::AbstractPropagator{M, K},
        direction::Type{<:Direction}) where {M, T, K <: AbstractFourierKernel{T, Nothing}}
    error("Propagation kernel has no defined wavelength")
end

function propagate!(u::ScalarField, p::AbstractPropagator{M, K},
        direction::Type{<:Direction}) where {M, T, K <: AbstractFourierKernel{T, Nothing}}
    _propagate_core!(u.data, p) do v
        apply_kernel!(v, p.kernel, u.lambdas, direction)
    end
    u
end

function propagate!(u::ScalarField, p::AbstractPropagator{M, K},
        direction::Type{<:Direction}) where {
        M, T, K <: AbstractFourierKernel{T, <:AbstractArray}}
    _propagate_core!(u.data, p) do v
        apply_kernel!(v, p.kernel, u.lambdas_collection, direction)
    end
    u
end

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
            new{T, Nothing, V}(s_vec, nothing, normalization_factor, k, z)
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

    function RSProp(u::AbstractArray{<:Complex, N},
            dx::Real,
            dy::Real,
            z::Real,
            lambdas::Union{Nothing, Tuple{Vararg{<:Real}}} = nothing) where {N}
        @assert N >= 2
        nx, ny = size(u)
        kernel = RSKernel(u, dx, dy, z, lambdas)
        u_tmp = similar(u, (2*nx-1, 2*ny-1))
        p_f = make_fft_plans(u_tmp, (1, 2))
        new{Static, typeof(kernel), typeof(u_tmp), typeof(p_f)}(kernel, p_f, u_tmp)
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
    u_view .= u
    p.p_f.ft * p.u_tmp
    apply_kernel_fn(p.u_tmp)
    p.p_f.ift * p.u_tmp
    @views u .= u_view
    u
end

function backpropagate!(u, p::AbstractPropagator, direction::Type{<:Direction})
    propagate!(u, p, reverse(direction))
end
