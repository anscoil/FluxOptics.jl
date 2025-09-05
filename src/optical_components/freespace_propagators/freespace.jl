include("abstract_kernel.jl")

abstract type AbstractPropagator{M, K, T} <: AbstractCustomComponent{M} end

function get_kernels(p::AbstractPropagator{M, <:AbstractKernel}) where {M}
    error("Not implemented")
end

function build_kernel_args(p::AbstractPropagator{M, <:AbstractKernel}) where {M}
    error("Not implemented")
end

function build_kernel_args(p::AbstractPropagator{M, <:AbstractKernel, T},
        λ::Real) where {M, T <: Real}
    (T(λ), build_kernel_args(p)...)
end

function build_kernel_args(p::AbstractPropagator{M, <:AbstractKernel},
        lambdas::AbstractArray) where {M}
    (lambdas, build_kernel_args(p)...)
end

function get_kernel_extra_key_params(p::AbstractPropagator{M, <:AbstractKernel}) where {M}
    error("Not implemented")
end

build_kernel_keys(p::AbstractPropagator{M, K, T}, λ::Real) where {M, K, T} = hash(T(λ))

function build_kernel_keys(p::AbstractPropagator{M, <:AbstractKernel},
        lambdas::AbstractArray) where {M}
    key_params = get_kernel_extra_key_params(p)
    (1 + length(key_params),
        [hash((λ, params...)) for (λ, params...) in zip(lambdas, key_params...)])
end

function build_kernel_args_dict(p::AbstractPropagator{M, <:AbstractKernel},
        λ::Real) where {M}
    build_kernel_args(p, λ)
end

function build_kernel_args_dict(p::AbstractPropagator{M, <:AbstractKernel},
        lambdas::AbstractArray) where {M}
    params = get_kernel_extra_key_params(p)
    n = length(params)
    (lambdas, params..., build_kernel_args(p)[(n + 1):end]...)
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
    kernel_key = build_kernel_keys(p, λ)
    kernel_args = build_kernel_args(p, λ)
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

function set_ds_out(p::AbstractPropagator, u::ScalarField, ::Type{<:Direction})
    u
end

function propagate!(u::ScalarField, p::AbstractPropagator{M, <:AbstractKernel{Nothing}},
        direction::Type{<:Direction}) where {M}
    kernels = get_kernels(p)
    kernel_args = build_kernel_args(p, u.lambdas)
    apply_kernel_fns = map(
        kernel -> (v,
            compute_kernel) -> apply_kernel!(
            v, kernel, direction, compute_kernel, kernel_args),
        kernels)
    _propagate_core!(apply_kernel_fns, u.data, p, direction)
    set_ds_out(p, u, direction)
end

function propagate!(u::ScalarField, p::AbstractPropagator{M, <:AbstractKernel{K}},
        direction::Type{<:Direction}) where {M, K <: AbstractArray}
    kernels = get_kernels(p)
    kernel_keys = build_kernel_keys(p, u.lambdas_collection)
    kernel_args = build_kernel_args_dict(p, u.lambdas_collection)
    apply_kernel_fns = map(
        kernel -> (v,
            compute_kernel) -> apply_kernel!(
            v, kernel, kernel_keys..., direction, compute_kernel, kernel_args),
        kernels)
    _propagate_core!(apply_kernel_fns, u.data, p, direction)
    set_ds_out(p, u, direction)
end

function backpropagate!(u, p::AbstractPropagator, direction::Type{<:Direction})
    propagate!(u, p, reverse(direction))
end

include("fourier_kernel.jl")
include("convolution_kernel.jl")
include("chirp_kernel.jl")
include("angular_spectrum.jl")
include("tilted_angular_spectrum.jl")
include("rayleigh_sommerfeld.jl")
include("collins_integral.jl")
include("fourier_lens.jl")
include("paraxial_propagation.jl")
