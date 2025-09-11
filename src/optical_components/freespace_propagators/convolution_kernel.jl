struct ConvolutionKernel{K, Nd, V, P, U} <: AbstractKernel{K, V}
    s_vec::V
    kernel_cache::Union{Nothing, LRU{UInt, K}}
    p_f::P
    u_plan::U

    function ConvolutionKernel(
            u::U,
            ns::NTuple{Nd, Integer},
            ds::NTuple{Nd, Real},
            cache_size::Integer
    ) where {N, Nd, U <: AbstractArray{<:Complex, N}}
        @assert Nd in (1, 2)
        @assert N >= Nd
        @assert cache_size >= 0
        F = adapt_dim(U, 1, real)
        s = [circshift((1 - nx):(nx - 1), nx) .* dx |> F for (nx, dx) in zip(ns, ds)]
        if Nd == 2
            nx, ny = ns
            s_vec = (; x = s[1], y = s[2]')
            u_plan = similar(u, (2*nx-1, 2*ny-1, size(u)[3:end]...))
        else
            nx = ns
            s_vec = (; x = s[1])
            u_plan = similar(u, (2*nx-1, size(u)[2:end]...))
        end
        V = typeof(s_vec)
        p_f = make_fft_plans(u_plan, Tuple(1:Nd))
        P = typeof(p_f)
        if iszero(cache_size)
            new{Nothing, Nd, V, P, U}(s_vec, nothing, p_f, u_plan)
        else
            K = adapt_dim(U, Nd)
            kernel_cache = LRU{UInt, K}(maxsize = cache_size)
            new{K, Nd, V, P, U}(s_vec, kernel_cache, p_f, u_plan)
        end
    end
end

function get_kernel_cache(kernel::ConvolutionKernel)
    kernel.kernel_cache
end

function get_kernel_vectors(kernel::ConvolutionKernel)
    kernel.s_vec
end

function transform_kernel!(kernel_val::Broadcast.Broadcasted,
        kernel::ConvolutionKernel{K, Nd}) where {K, Nd}
    fft!(Broadcast.materialize(kernel_val), Tuple(1:Nd))
end

function transform_kernel!(kernel_val::AbstractArray,
        kernel::ConvolutionKernel{K, Nd}) where {K, Nd}
    fft!(kernel_val, Tuple(1:Nd))
end
