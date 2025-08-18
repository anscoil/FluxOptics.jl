struct FourierKernel{T, K, V, P}
    f_vec::V
    kernel_cache::Union{Nothing, LRU{T, K}}
    p_f::P

    function FourierKernel(
            u::U,
            ns::NTuple{Nd, Integer},
            ds::NTuple{Nd, Real},
            cache_size::Integer
    ) where {T <: Real, N, Nd, U <: AbstractArray{Complex{T}, N}}
        @assert Nd in (1, 2)
        @assert N >= Nd
        @assert cache_size >= 0
        F = adapt_dim(U, 1, real)
        fs = [fftfreq(nx, 1/dx) |> F for (nx, dx) in zip(ns, ds)]
        f_vec = Nd == 2 ? (; x = fs[1], y = fs[2]') : (x = fs[1],)
        V = typeof(f_vec)
        A_plan = similar(u)
        p_f = make_fft_plans(A_plan, Tuple(1:Nd))
        P = typeof(p_f)
        if iszero(cache_size)
            new{T, Nothing, V, P}(f_vec, nothing, p_f)
        else
            K = adapt_dim(U, Nd)
            kernel_cache = LRU{T, K}(maxsize = cache_size)
            new{T, K, V, P}(f_vec, kernel_cache, p_f)
        end
    end
end

function apply_kernel!(u::AbstractArray, kernel::FourierKernel{T, Nothing},
        direction::Type{<:Direction}, compute_kernel::F, lambdas, args::A) where {T, F, A}
    @. u *= kernel_direction(compute_kernel(kernel.f_vec..., lambdas, args...), direction)
end

function fill_kernel_cache(kernel::FourierKernel{T, K}, kernel_key::T,
        compute_kernel::F, λ::Real, args::A) where {T, K <: AbstractArray, F, A}
    kernel_cache = kernel.kernel_cache
    kernel_val = @. compute_kernel(kernel.f_vec..., λ, args...)
    kernel_cache[kernel_key] = kernel_val
    kernel_val
end

function apply_kernel!(u::AbstractArray, kernel::FourierKernel{T, K},
        direction::Type{<:Direction}, compute_kernel::F, λ::Real,
        args::A) where {T, K <: AbstractArray, F, A}
    kernel_key = T(λ)
    kernel_cache = kernel.kernel_cache
    if haskey(kernel_cache, kernel_key)
        @. u *= kernel_direction(kernel_cache[kernel_key], direction)
    else
        kernel_val = fill_kernel_cache(kernel, kernel_key, compute_kernel, λ, args)
        @. u *= kernel_direction(kernel_val, direction)
    end
    u
end

function apply_kernel!(u::AbstractArray, kernel::FourierKernel{T, K},
        direction::Type{<:Direction}, compute_kernel::F, lambdas::AbstractArray,
        args::A) where {T, K <: AbstractArray, F, A}
    n = length(kernel.f_vec)
    inds = CartesianIndices(size(u)[(n + 1):end])
    for i in inds
        apply_kernel!(
            @view(u[.., i]), kernel, direction, compute_kernel, lambdas[i], args)
    end
    u
end
