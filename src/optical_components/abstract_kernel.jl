abstract type AbstractKernel{K, V} end

function get_kernel_cache(kernel::AbstractKernel)
    error("Not implemented")
end

function get_kernel_vectors(kernel::AbstractKernel)
    error("Not implemented")
end

function transform_kernel!(kernel_val, kernel::AbstractKernel)
    error("Not implemented")
end

function fill_kernel_cache(kernel::AbstractKernel{K},
        kernel_key::UInt,
        compute_kernel::F,
        args::A) where {K <: AbstractArray, F, A}
    kernel_cache = get_kernel_cache(kernel)
    k_vec = get_kernel_vectors(kernel)
    kernel_val = @. compute_kernel(k_vec..., args...)
    transform_kernel!(kernel_val, kernel)
    kernel_cache[kernel_key] = kernel_val
    kernel_val
end

function apply_kernel!(u::AbstractArray, kernel::AbstractKernel{Nothing},
        direction::Type{<:Direction}, compute_kernel::F, args::A) where {F, A}
    k_vec = get_kernel_vectors(kernel)
    kernel_val = Base.broadcasted(compute_kernel, k_vec..., args...)
    transform_kernel!(kernel_val, kernel)
    @. u *= kernel_direction(kernel_val, direction)
end

function apply_kernel!(u::AbstractArray,
        kernel::AbstractKernel{K},
        kernel_key::UInt,
        direction::Type{<:Direction}
) where {K <: AbstractArray}
    kernel_cache = get_kernel_cache(kernel)
    kernel_val = kernel_cache[kernel_key]
    @. u *= kernel_direction(kernel_val, direction)
end

function apply_kernel!(u::AbstractArray,
        kernel::AbstractKernel{K},
        kernel_key::UInt,
        direction::Type{<:Direction},
        compute_kernel::F,
        args::A
) where {K <: AbstractArray, F, A}
    kernel_cache = get_kernel_cache(kernel)
    if haskey(kernel_cache, kernel_key)
        apply_kernel!(u, kernel, kernel_key, direction)
    else
        kernel_val = fill_kernel_cache(
            kernel, kernel_key, compute_kernel, args)
        @. u *= kernel_direction(kernel_val, direction)
    end
    u
end

function apply_kernel!(u::AbstractArray,
        kernel::AbstractKernel{K},
        kernel_keys::AbstractArray{UInt},
        direction::Type{<:Direction},
        compute_kernel::F,
        args::A,
        nhead::Integer = 1
) where {K <: AbstractArray, F, A}
    k_vec = get_kernel_vectors(kernel)
    n = length(k_vec)
    inds = CartesianIndices(size(u)[(n + 1):end])
    args_head = args[1:nhead]
    args_tail = args[(nhead + 1):end]
    for i in inds
        args = (ntuple(k -> args_head[k][i], nhead)..., args_tail...)
        kernel_key = kernel_keys[i]
        apply_kernel!(
            @view(u[.., i]), kernel, kernel_key, direction, compute_kernel, args)
    end
    u
end
