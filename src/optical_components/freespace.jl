using EllipsisNotation
using LRUCache

function get_kernels(p::AbstractPropagator{M, <:AbstractKernel}) where {M}
    error("Not implemented")
end

function build_kernel_key_args(p::AbstractPropagator{M, <:AbstractKernel}, args...
) where {M}
    error("Not implemented")
end

function build_kernel_args(
        p::AbstractPropagator{M, <:AbstractKernel{Nothing}},
        u::ScalarField) where {M}
    error("Not implemented")
end

function build_kernel_key_args(
        p::AbstractPropagator{M, <:AbstractKernel{K}},
        u::ScalarField) where {M, K <: AbstractArray}
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
    kernels = get_kernels(p)
    kernel_key, kernel_args = build_kernel_key_args(p, λ)
    apply_kernel_fns = map(
        kernel -> (v,
            compute_kernel) -> apply_kernel!(
            v, kernel, kernel_key, direction, compute_kernel, kernel_args),
        kernels)
    _propagate_core!(apply_kernel_fns, u, p, direction)
    u
end

function propagate!(u::AbstractArray, p::AbstractPropagator{M, <:AbstractKernel{K}},
        direction::Type{<:Direction}) where {M, K <: AbstractArray}
    kernels = get_kernels(p)
    kernel_caches = map(get_kernel_cache, kernels)
    (all(!isnothing, kernel_caches) && all((==)(1) ∘ length, kernel_caches)) ||
        error("Propagation kernel should hold exactly one wavelength")
    kernel_keys = map(first ∘ keys, kernel_caches)
    kernel_key = kernel_keys[1]
    all((==)(kernel_key), kernel_keys) || error("All kernel keys must be equal")
    apply_kernel_fns = map(
        kernel -> (v, compute_kernel) -> apply_kernel!(v, kernel, kernel_key, direction),
        kernels)
    _propagate_core!(apply_kernel_fns, u, p, direction)
    u
end

function propagate!(u::ScalarField, p::AbstractPropagator{M, <:AbstractKernel{Nothing}},
        direction::Type{<:Direction}) where {M}
    kernels = get_kernels(p)
    kernel_args = build_kernel_args(p, u)
    apply_kernel_fns = map(
        kernel -> (v,
            compute_kernel) -> apply_kernel!(
            v, kernel, direction, compute_kernel, kernel_args),
        kernels)
    _propagate_core!(apply_kernel_fns, u.data, p, direction)
    u
end

function propagate!(u::ScalarField, p::AbstractPropagator{M, <:AbstractKernel{K}},
        direction::Type{<:Direction}) where {M, K <: AbstractArray}
    kernels = get_kernels(p)
    kernel_keys, kernel_args = build_kernel_key_args(p, u)
    apply_kernel_fns = map(
        kernel -> (v,
            compute_kernel) -> apply_kernel!(
            v, kernel, kernel_keys, direction, compute_kernel, kernel_args),
        kernels)
    _propagate_core!(apply_kernel_fns, u.data, p, direction)
    u
end

function backpropagate!(u, p::AbstractPropagator, direction::Type{<:Direction})
    propagate!(u, p, reverse(direction))
end

include("fourier_kernel.jl")
include("convolution_kernel.jl")
include("chirp_kernel.jl")
include("angular_spectrum.jl")
include("rayleigh_sommerfeld.jl")
include("collins_integral.jl")
include("fourier_lens.jl")
