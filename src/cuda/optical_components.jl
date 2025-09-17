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

function OpticalComponents.make_nufft_plan(
        u::CuArray{Complex{T}, 2}, ns::Tuple{Integer, Integer},
        s::Tuple{AbstractMatrix, AbstractMatrix}, type::Integer,
        isign::Integer, eps::Real) where {T <: Real}
    p_nft = cufinufft_makeplan(type, [ns...], isign, 1, eps; dtype = T)
    cufinufft_setpts!(p_nft, s...)
    p_nft
end

function OpticalComponents.exec_nufft_plan!(
        p, u::CuArray{Complex{T}, 2}) where {T <: Real}
    nx, ny = size(u)
    u_in = reshape(u, (nx, ny, 1))
    u_out = reshape(u, (:, 1))
    if p.type == 2
        cufinufft_exec!(p, u_in, u_out)
    end
    if p.type == 1
        cufinufft_exec!(p, u_out, u_in)
    end
    if p.type == 3
        cufinufft_exec!(p, u_out, u_out)
    end
    u
end
