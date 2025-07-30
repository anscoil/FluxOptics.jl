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

abstract type AbstractFourierKernel{T, K} end

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

function apply_kernel!(u::AbstractArray, as_k::ASKernel{T, Nothing},
        λ::Real, direction::Type{<:Direction}) where {T}
    fx, fy = as_k.f_vec.x, as_k.f_vec.y
    @. u *= kernel_direction(as_kernel(fx, fy', T(λ), as_k.z), direction)
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
end

function apply_kernel!(u::ScalarField{U, A},
        kernel::AbstractFourierKernel{T, K},
        direction::Type{<:Direction}) where {U, T, A <: AbstractArray, K <: AbstractArray}
    inds = CartesianIndices(size(u.data)[3:end])
    for i in inds
        apply_kernel!(@view(u.data[:, :, i]), kernel, u.lambdas_collection[i], direction)
    end
    u
end

function apply_kernel!(u::ScalarField{U, T},
        kernel::AbstractFourierKernel,
        direction::Type{<:Direction}) where {U, T <: Real}
    apply_kernel!(u.data, kernel, u.lambdas, direction)
end

function apply_kernel!(u::ScalarField{U, A},
        as_k::ASKernel{T, Nothing},
        direction::Type{<:Direction}) where {U, T, A <: AbstractArray}
    fx, fy = as_k.f_vec.x, as_k.f_vec.y
    @. u.data *= kernel_direction(as_kernel(fx, fy', u.lambdas, as_k.z), direction)
    u
end

struct ASProp{M, K, P} <: AbstractPropagator{M}
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

function propagate!(u::AbstractArray, p::ASProp, λ::Real, direction::Type{<:Direction})
    p.p_f.ft * u
    apply_kernel!(u, p.kernel, λ, direction)
    p.p_f.ift * u
end

function propagate!(u::AbstractArray, p::ASProp{M, K},
        direction::Type{<:Direction}) where {M, T, K <: ASKernel{T, <:AbstractArray}}
    kernel_cache = p.kernel.kernel_cache
    length(kernel_cache) == 1 || error("Propagation kernel should hold only one wavelength")
    λ = first(keys(kernel_cache))
    propagate!(u, p, λ, direction)
end

function propagate!(u::AbstractArray, p::ASProp{M, K},
        direction::Type{<:Direction}) where {M, T, K <: ASKernel{T, Nothing}}
    error("Propagation kernel has no defined wavelength")
end

function propagate!(u::ScalarField, p::ASProp, direction::Type{<:Direction})
    p.p_f.ft * u.data
    apply_kernel!(u, p.kernel, direction)
    p.p_f.ift * u.data
    u
end

struct RSProp{M, K, T, U, P} <: AbstractPropagator{M}
    p_ker::K
    u_tmp::U
    p_f::P

    function RSProp(U::Type{<:AbstractArray{Complex{T}, N}},
            dims::NTuple{N, Integer}, dx::Real, dy::Real, λ::Real, z::Real
    ) where {N, T <: Real}
        @assert N >= 2
        K = adapt_dim(U, 2)
        nx, ny = dims
        Nx, Ny = 2*nx-1, 2*ny-1
        x_vec = circshift((1 - nx):(nx - 1), nx) .* dx
        y_vec = circshift((1 - ny):(ny - 1), ny) .* dy
        k = 2*π/λ
        r_vec = @. sqrt(x_vec^2 + (y_vec')^2 + z^2)
        p_ker = (@. (dx*dy/2π)*(exp(im*k*r_vec)/r_vec)*(z/r_vec)*(1/r_vec-im*k)) |> K
        fft!(p_ker)
        A_plan = U(undef, (Nx, Ny, dims[3:end]...))
        p_f = make_fft_plans(A_plan, (1, 2))
        P = typeof(p_f)
        new{Static, K, T, U, P}(p_ker, A_plan, p_f)
    end

    function RSProp(u::U,
            dx::Real, dy::Real, λ::Real, z::Real
    ) where {U <: AbstractArray{<:Complex}}
        RSProp(U, size(u), dx, dy, λ, z)
    end
end

function propagate!(u::AbstractArray, rs_prop::RSProp, direction::Type{<:Direction})
    nx, ny = size(u)
    rs_prop.u_tmp .= 0
    u_view = @view rs_prop.u_tmp[1:nx, 1:ny, ..]
    u_view .= u
    rs_prop.p_f.ft * rs_prop.u_tmp
    apply_kernel!(rs_prop.u_tmp, rs_prop, direction)
    rs_prop.p_f.ift * rs_prop.u_tmp
    @views u .= u_view
    u
end

function backpropagate!(u, p::AbstractPropagator, direction::Type{<:Direction})
    propagate!(u, p, reverse(direction))
end
