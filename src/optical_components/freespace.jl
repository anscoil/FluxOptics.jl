using EllipsisNotation
using LRUCache

function kernel_direction(kernel, ::Type{Forward})
    kernel
end

function kernel_direction(kernel, ::Type{Backward})
    conj(kernel)
end

function apply_kernel!(u::AbstractArray, as_k::AbstractFourierKernel, lambdas,
        direction::Type{<:Direction})
    error("Not implemented")
end

function apply_kernel!(u::AbstractArray,
        kernel::AbstractFourierKernel{T, K},
        lambdas::AbstractArray,
        direction::Type{<:Direction}) where {T, K <: AbstractArray}
    inds = CartesianIndices(size(u)[3:end])
    for i in inds
        apply_kernel!(@view(u[:, :, i]), kernel, lambdas[i], direction)
    end
    u
end

function get_kernel_cache(p::AbstractPropagator{M, K}) where {
        M, K <: AbstractFourierKernel}
    error("Not Implemented")
end

function _propagate_core!(apply_kernel_fn, u::AbstractArray, p::AbstractPropagator)
    error("Not Implemented")
end

function propagate!(u::AbstractArray, p::AbstractPropagator{M, K}, λ::Real,
        direction::Type{<:Direction}
) where {M, T, K <: AbstractFourierKernel{T}}
    _propagate_core!(u, p) do v
        apply_kernel!(v, p.kernel, T(λ), direction)
    end
    u
end

function propagate!(u::AbstractArray, p::AbstractPropagator{M, K},
        direction::Type{<:Direction}) where {
        M, T, K <: AbstractFourierKernel{T, <:AbstractArray}}
    kernel_cache = get_kernel_cache(p)
    length(kernel_cache) == 1 ||
        error("Propagation kernel should hold exactly one wavelength")
    λ = first(keys(kernel_cache))
    propagate!(u, p, λ, direction)
end

function propagate!(u::AbstractArray, p::AbstractPropagator{M, K},
        direction::Type{<:Direction}) where {M, T, K <: AbstractFourierKernel{T, Nothing}}
    error("Propagation kernel has no defined wavelength")
end

function propagate!(u::ScalarField, p::AbstractPropagator{M, K},
        direction::Type{<:Direction}) where {M, T, K <: AbstractFourierKernel{T, Nothing}}
    _propagate_core!(u.data, p) do v
        apply_kernel!(v, p.kernel, u.lambdas, direction)
    end
    u
end

function propagate!(u::ScalarField, p::AbstractPropagator{M, K},
        direction::Type{<:Direction}) where {
        M, T, K <: AbstractFourierKernel{T, <:AbstractArray}}
    _propagate_core!(u.data, p) do v
        apply_kernel!(v, p.kernel, u.lambdas_collection, direction)
    end
    u
end

function backpropagate!(u, p::AbstractPropagator, direction::Type{<:Direction})
    propagate!(u, p, reverse(direction))
end

include("angular_spectrum.jl")
include("rayleigh_sommerfeld.jl")
include("fourier_lens.jl")
