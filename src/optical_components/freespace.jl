using EllipsisNotation
using LRUCache

function kernel_direction(kernel, ::Type{Forward})
    kernel
end

function kernel_direction(kernel, ::Type{Backward})
    conj(kernel)
end

function get_kernel(p::AbstractPropagator{M, <:AbstractKernel}) where {M}
    error("Not implemented")
end

function build_kernel_key_args(p::AbstractPropagator{M, <:AbstractKernel}, args...
) where {M}
    error("Not implemented")
end

function build_kernel_args(
        p::AbstractPropagator{M, <:AbstractKernel{Nothing}},
        u:: ScalarField) where {M}
    error("Not implemented")
end

function build_kernel_key_args(
        p::AbstractPropagator{M, <:AbstractKernel{K}},
        u:: ScalarField) where {M, K <: AbstractArray}
    error("Not implemented")
end

function _propagate_core!(
         apply_kernel_fn!::F,
         u::AbstractArray,
         p::AbstractPropagator{M, <:AbstractKernel}) where {F, M}
    error("Not implemented")
end

function propagate!(u::AbstractArray, p::AbstractPropagator{M, <:AbstractKernel{K}},
         λ::Real, direction::Type{<:Direction}) where {M, K <: AbstractArray}
    kernel = get_kernel(p)
    kernel_key, kernel_args = build_kernel_key_args(p, λ)
    _propagate_core!(u, p) do v, compute_kernel
        apply_kernel!(v, kernel, kernel_key, direction, compute_kernel, kernel_args)
    end
    u
end

function propagate!(u::AbstractArray, p::AbstractPropagator{M, <:AbstractKernel{K}},
        direction::Type{<:Direction}) where {M, K <: AbstractArray}
    kernel = get_kernel(p)
    kernel_cache = get_kernel_cache(kernel)
    (!isnothing(kernel_cache) && length(kernel_cache) == 1) ||
        error("Propagation kernel should hold exactly one wavelength")
    kernel_key = first(keys(kernel_cache))
    _propagate_core!(u, p) do v, compute_kernel
        apply_kernel!(v, kernel, kernel_key, direction)
    end
    u
end

function propagate!(u::ScalarField, p::AbstractPropagator{M, <:AbstractKernel{Nothing}},
        direction::Type{<:Direction}) where {M}
    kernel = get_kernel(p)
    kernel_args = build_kernel_args(p, u)
    _propagate_core!(u.data, p) do v, compute_kernel
        apply_kernel!(v, kernel, direction, compute_kernel, kernel_args)
    end
    u
end

function propagate!(u::ScalarField, p::AbstractPropagator{M, <:AbstractKernel{K}},
        direction::Type{<:Direction}) where {M, K <: AbstractArray}
    kernel = get_kernel(p)
    kernel_keys, kernel_args = build_kernel_key_args(p, u)
    _propagate_core!(u.data, p) do v, compute_kernel
        apply_kernel!(v, kernel, kernel_keys, direction, compute_kernel, kernel_args)
    end
    u
end

function backpropagate!(u, p::AbstractPropagator, direction::Type{<:Direction})
    propagate!(u, p, reverse(direction))
end

include("fourier_kernel.jl")
include("convolution_kernel.jl")
include("angular_spectrum.jl")
include("rayleigh_sommerfeld.jl")
include("fourier_lens.jl")
