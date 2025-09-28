include("abstract_kernel.jl")

abstract type AbstractPropagator{M, K, T} <: AbstractCustomComponent{M} end

function get_kernels(p::AbstractPropagator{M, <:AbstractKernel}) where {M}
    error("Not implemented")
end

function get_data(p::AbstractPropagator{M, <:AbstractKernel}) where {M}
    first(get_kernels(p))
end

function build_kernel_key_args(p::AbstractPropagator{M, <:AbstractKernel},
                               u::ScalarField) where {M}
    error("Not implemented")
end

function build_kernel_args(p::AbstractPropagator{M, <:AbstractKernel},
                           u::ScalarField) where {M}
    error("Not implemented")
end

function _propagate_core!(apply_kernel_fn!::F,
                          u::ScalarField,
                          p::AbstractPropagator{M, <:AbstractKernel}) where {F, M}
    error("Not implemented")
end

set_ds_out(p::AbstractPropagator, u::ScalarField, ::Type{<:Direction}) = u

function propagate!(u::ScalarField,
                    p::AbstractPropagator{M, <:AbstractKernel{Nothing}},
                    direction::Type{<:Direction}) where {M}
    kernels = get_kernels(p)
    kernel_key_args = map(f -> f(false), build_kernel_key_args(p, u))
    kernel_args = build_kernel_args(p, u)
    all_args = (kernel_key_args..., kernel_args...)
    apply_kernel_fns = map(kernel -> (v,
                                      compute_kernel) -> apply_kernel!(v, kernel,
                                                                       compute_kernel,
                                                                       all_args, direction),
                           kernels)
    _propagate_core!(apply_kernel_fns, u, p, direction)
    set_ds_out(p, u, direction)
end

function propagate!(u::ScalarField,
                    p::AbstractPropagator{M, <:AbstractKernel{K}},
                    direction::Type{<:Direction}) where {M, K <: AbstractArray}
    kernels = get_kernels(p)
    kernel_key_args = map(f -> f(true), build_kernel_key_args(p, u))
    kernel_args = build_kernel_args(p, u)
    apply_kernel_fns = map(kernel -> (v,
                                      compute_kernel) -> apply_kernel!(v,
                                                                       kernel,
                                                                       compute_kernel,
                                                                       kernel_key_args,
                                                                       kernel_args,
                                                                       direction),
                           kernels)
    _propagate_core!(apply_kernel_fns, u, p, direction)
    set_ds_out(p, u, direction)
end

function backpropagate!(u, p::AbstractPropagator, direction::Type{<:Direction})
    propagate!(u, p, reverse(direction))
end

include("fourier_kernel.jl")
include("convolution_kernel.jl")
include("chirp_kernel.jl")
include("angular_spectrum.jl")
include("angular_spectrum_rotation.jl")
include("shift_kernel.jl")
include("rayleigh_sommerfeld.jl")
include("collins_integral.jl")
include("fourier_lens.jl")
include("paraxial_propagation.jl")
