module CUDAExt

using AbstractFFTs
using CUDA
using ..FFTutils
using ..OpticalComponents
using ..ProximalOperators
using ..Fields

function CUDA.cu(u::ScalarField)
    ScalarField(cu(u.data), u.ds, u.lambdas)
end

function compute_thread_config()
    props = CUDA.device()
    max_threads = CUDA.attribute(props, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)

    base = floor(Int, sqrt(max_threads))
    threads_x = base
    threads_y = div(max_threads, threads_x)

    return (threads_x, threads_y)
end

function Base.unique(x::CuArray)
    unique(Array(x))
end

function FFTutils.make_fft_plans(
        u::U, dims::NTuple{N, Integer}) where {N, U <: CuArray{<:Complex}}
    p_ft = plan_fft!(u, dims)
    p_ift = plan_ifft!(u, dims)
    (; ft = p_ft, ift = p_ift)
end

include("optical_components.jl")

end
