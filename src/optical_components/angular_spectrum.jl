function as_kernel(fx::T, fy::T, λ::T, z::T) where {T <: AbstractFloat}
    f² = complex(1/λ^2)
    exp(im*T(2)*π*z*sqrt(f² - fx^2 - fy^2))
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

function get_kernel_cache(p::ASProp)
    p.kernel.kernel_cache
end

function _propagate_core!(apply_kernel_fn, u::AbstractArray, p::ASProp)
    p.p_f.ft * u
    apply_kernel_fn(u)
    p.p_f.ift * u
end
