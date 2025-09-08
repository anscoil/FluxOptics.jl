module CUDAExt

using AbstractFFTs
using CUDA
using ..FFTutils
using ..OpticalComponents
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

function kernel_phase_gradient!(∂ϕ, ∂u, u, s)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i > size(∂ϕ, 1) || j > size(∂ϕ, 2)
        return
    end

    acc = zero(eltype(∂ϕ))

    for k in 1:size(∂u, 3)
        a = ∂u[i, j, k]
        b = u[i, j, k]
        acc += imag(a * conj(b))
    end

    ∂ϕ[i, j] = s*acc
    return
end

function OpticalComponents.compute_phase_gradient!(∂ϕ::CuArray{<:Real, Nd},
        u_saved, ∂u::ScalarField, direction) where {Nd}
    nx, ny = size(u_saved)
    @assert size(∂u) == size(u_saved)
    @assert size(∂ϕ, 1) == nx
    @assert size(∂ϕ, 2) == ny
    s = sign(direction)
    nz = prod(size(∂u)[3:end])

    tx, ty = compute_thread_config()
    threads = (tx, ty)
    blocks = (
        cld(size(∂ϕ, 1), tx),
        cld(size(∂ϕ, 2), ty)
    )

    @cuda threads=threads blocks=blocks kernel_phase_gradient!(
        ∂ϕ, reshape(∂u.data, (nx, ny, nz)), reshape(u_saved, (nx, ny, nz)), s)

    ∂ϕ
end

end
