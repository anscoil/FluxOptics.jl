struct ChirpKernel{K, V} <: AbstractKernel{K, V}
    s_vec::V
    kernel_cache::Union{Nothing, LRU{UInt, K}}

    function ChirpKernel(u::U,
                         ns::NTuple{Nd, Integer},
                         ds::NTuple{Nd, Real},
                         cache_size::Integer) where {N, Nd,
                                                     U <: AbstractArray{<:Complex, N}}
        @assert Nd in (1, 2)
        @assert N >= Nd
        @assert cache_size >= 0
        F = similar(U, real, 1)
        s = [((0:(2 * (nx - 1))) .- (nx-1)/2) .* dx |> F for (nx, dx) in zip(ns, ds)]
        s_vec = Nd == 2 ? s_vec = (; x = s[1], y = s[2]') : (; x = s[1])
        V = typeof(s_vec)
        if iszero(cache_size)
            new{Nothing, V}(s_vec, nothing)
        else
            K = similar(U, Nd)
            kernel_cache = LRU{UInt, K}(maxsize = cache_size)
            new{K, V}(s_vec, kernel_cache)
        end
    end
end

function get_kernel_cache(kernel::ChirpKernel)
    kernel.kernel_cache
end

function get_kernel_vectors(kernel::ChirpKernel)
    kernel.s_vec
end

function transform_kernel!(kernel_val, kernel::ChirpKernel)
    kernel_val
end
