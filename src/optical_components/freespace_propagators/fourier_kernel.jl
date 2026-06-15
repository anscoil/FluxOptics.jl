struct FourierKernel{K, V, T, P} <: AbstractKernel{K, V}
    f_vec::V
    kernel_cache::Union{Nothing, LRU{UInt, K}}
    p_f::P
    nrm_f::T

    function FourierKernel(u::U,
                           ns::NTuple{Nd, Integer},
                           ds::NTuple{Nd, Real},
                           cache_size::Integer,
                           kernel_dim::Integer = Nd;
                           normalize::Bool = true
                           ) where {N, Nd, U <: AbstractArray{<:Complex, N}}
        @assert Nd in (1, 2)
        @assert N >= Nd
        @assert cache_size >= 0
        F = similar(U, real, 1)
        fs = [fftfreq(nx, 1/dx) |> F for (nx, dx) in zip(ns, ds)]
        f_vec = Nd == 2 ? (; x = fs[1], y = fs[2]') : (; x = fs[1])
        V = typeof(f_vec)
        u_plan = similar(u)
        p_f, nrm_f = make_fft_plans(u_plan, Tuple(1:Nd); normalize)
        P = typeof(p_f)
        T = typeof(nrm_f)
        if iszero(cache_size)
            new{Nothing, V, T, P}(f_vec, nothing, p_f, nrm_f)
        else
            K = similar(U, kernel_dim)
            kernel_cache = LRU{UInt, K}(maxsize = cache_size)
            new{K, V, T, P}(f_vec, kernel_cache, p_f, nrm_f)
        end
    end
end

Functors.@leaf LRU

Functors.@functor FourierKernel (kernel_cache,)

get_data(kernel::FourierKernel) = kernel.kernel_cache

function get_kernel_cache(kernel::FourierKernel)
    kernel.kernel_cache
end

function get_kernel_vectors(kernel::FourierKernel)
    kernel.f_vec
end

function transform_kernel!(kernel_val, kernel::FourierKernel)
    kernel_val
end
