struct FourierKernel{K, V, P} <: AbstractKernel{K, V, 1}
    f_vec::V
    kernel_cache::Union{Nothing, LRU{UInt, K}}
    p_f::P

    function FourierKernel(
            u::U,
            ns::NTuple{Nd, Integer},
            ds::NTuple{Nd, Real},
            cache_size::Integer
    ) where {N, Nd, U <: AbstractArray{<:Complex, N}}
        @assert Nd in (1, 2)
        @assert N >= Nd
        @assert cache_size >= 0
        F = adapt_dim(U, 1, real)
        fs = [fftfreq(nx, 1/dx) |> F for (nx, dx) in zip(ns, ds)]
        f_vec = Nd == 2 ? (; x = fs[1], y = fs[2]') : (; x = fs[1])
        V = typeof(f_vec)
        u_plan = similar(u)
        p_f = make_fft_plans(u_plan, Tuple(1:Nd))
        P = typeof(p_f)
        if iszero(cache_size)
            new{Nothing, V, P}(f_vec, nothing, p_f)
        else
            K = adapt_dim(U, Nd)
            kernel_cache = LRU{UInt, K}(maxsize = cache_size)
            new{K, V, P}(f_vec, kernel_cache, p_f)
        end
    end
end

function get_kernel_cache(kernel::FourierKernel)
    kernel.kernel_cache
end

function get_kernel_vectors(kernel::FourierKernel)
    kernel.f_vec
end

function transform_kernel!(kernel_val, kernel::FourierKernel)
    kernel_val
end
