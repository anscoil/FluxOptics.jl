function D!(DA::AbstractArray{T, 3}, A::AbstractArray{T, 2}) where {T <: Real}
    @assert size(DA, 1) == size(A, 1)
    @assert size(DA, 2) == size(A, 2)
    @assert size(DA, 3) == 2

    @views DA[1:(end - 1), :, 1] .= A[2:end, :] .- A[1:(end - 1), :]
    @views DA[:, 1:(end - 1), 2] .= A[:, 2:end] .- A[:, 1:(end - 1)]

    # Periodic boundary conditions
    @views DA[end, :, 1] .= A[1, :] .- A[end, :]
    @views DA[:, end, 2] .= A[:, 1] .- A[:, end]

    DA
end

function Dᵀ!(A::AbstractArray{T, 2}, DA::AbstractArray{T, 3}) where {T <: Real}
    @assert size(DA, 1) == size(A, 1)
    @assert size(DA, 2) == size(A, 2)
    @assert size(DA, 3) == 2

    @views A[2:end, :] .= DA[1:(end - 1), :, 1] .- DA[2:end, :, 1]
    @views A[:, 2:end] .+= DA[:, 1:(end - 1), 2] .- DA[:, 2:end, 2]

    @views A[1, :] .= DA[end, :, 1] .- DA[1, :, 1]
    @views A[:, 1] .+= DA[:, end, 2] .- DA[:, 1, 2]
    A
end

struct TVProx <: AbstractProximalOperator
    λ::Float64
    n_iter::Int
    function TVProx(λ::Real, n_iter::Integer)
        new(λ, n_iter)
    end
end

function init(prox::TVProx, x::AbstractArray)
    p = similar(x, (size(x)..., ndims(x)))
    opt = Optimisers.setup(Fista(0.25), p)
    ∂p = similar(p)
    y = similar(x)
    (opt, p, ∂p, y)
end

function normalize_if_greater_than_one!(x::AbstractArray)
    n = norm(x)
    if n > 1
        x ./= norm(x)
    end
    x
end

function apply!(prox::TVProx, (opt, p, ∂p, y), y0::AbstractArray{T}) where {T <: Real}
    p .= 0
    opt.state[2] .= 0
    opt.state[3] .= 0
    opt.state = (T(1), opt.state[2:3]...)
    for i in 1:prox.n_iter
        Dᵀ!(y, p)
        @. y = prox.λ*y - y0
        D!(∂p, y)
        Optimisers.update!(opt, p, ∂p)
        foreach(normalize_if_greater_than_one!, eachslice(p, dims = ndims(p)))
    end
    Dᵀ!(y, p)
    @. y0 = y0 - prox.λ*y
end
