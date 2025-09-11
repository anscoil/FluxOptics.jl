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
        kernel_args::A) where {K <: AbstractArray, F, A}
    kernel_cache = get_kernel_cache(kernel)
    k_vec = get_kernel_vectors(kernel)
    kernel_val = @. compute_kernel(k_vec..., kernel_args...)
    transform_kernel!(kernel_val, kernel)
    kernel_cache[kernel_key] = kernel_val
    kernel_val
end

function apply_kernel!(u::AbstractArray, kernel::AbstractKernel{Nothing},
        compute_kernel::F, kernel_args::A, direction::Type{<:Direction}) where {F, A}
    k_vec = get_kernel_vectors(kernel)
    kernel_val = transform_kernel!(
        Base.broadcasted(compute_kernel, k_vec..., kernel_args...), kernel)
    @. u *= conj_direction(kernel_val, direction)
end

function apply_kernel!(u::AbstractArray,
        kernel::AbstractKernel{K},
        kernel_key::UInt,
        direction::Type{<:Direction}
) where {K <: AbstractArray}
    kernel_cache = get_kernel_cache(kernel)
    kernel_val = kernel_cache[kernel_key]
    @. u *= conj_direction(kernel_val, direction)
end

function apply_kernel!(u::AbstractArray,
        kernel::AbstractKernel{K},
        kernel_key::UInt,
        compute_kernel::F,
        kernel_args::A,
        direction::Type{<:Direction}
) where {K <: AbstractArray, F, A}
    kernel_cache = get_kernel_cache(kernel)
    if haskey(kernel_cache, kernel_key)
        apply_kernel!(u, kernel, kernel_key, direction)
    else
        kernel_val = fill_kernel_cache(
            kernel, kernel_key, compute_kernel, kernel_args)
        @. u *= conj_direction(kernel_val, direction)
    end
    u
end

function apply_kernel!(u::AbstractArray,
        kernel::AbstractKernel{K},
        compute_kernel::F,
        kernel_key_args::A,
        kernel_args::B,
        direction::Type{<:Direction}
) where {K <: AbstractArray, F, A, B}
    Nd = length(get_kernel_vectors(kernel))
    inds = CartesianIndices(size(u)[(Nd + 1):end])
    for (i, key_args...) in bzip(inds, kernel_key_args...)
        kernel_key = hash(key_args)
        all_args = (key_args..., kernel_args...)
        apply_kernel!(
            @view(u[.., i]), kernel, kernel_key, compute_kernel, all_args, direction)
    end
    u
end
