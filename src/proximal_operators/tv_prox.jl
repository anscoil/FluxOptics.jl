function finite_difference_forward!(DA::AbstractArray{T},
        A::AbstractArray{T, Nd}, dim::Integer) where {Nd, T <: Real}
    @assert dim in 1:Nd
    n = size(A, dim)
    DA_v = view(DA, ntuple(k -> k == dim ? (1:(n - 1)) : axes(A, k), Nd)..., dim)
    A2_v = view(A, ntuple(k -> k == dim ? (2:n) : axes(A, k), Nd)...)
    A1_v = view(A, ntuple(k -> k == dim ? (1:(n - 1)) : axes(A, k), Nd)...)
    @. DA_v = A2_v - A1_v
    DA
end

function finite_difference_backward!(A::AbstractArray{T, Nd},
        DA::AbstractArray{T}, dim::Integer, add = false) where {Nd, T <: Real}
    @assert dim in 1:Nd
    n = size(A, dim)
    A_v = view(A, ntuple(k -> k == dim ? (2:n) : axes(A, k), Nd)...)
    DA1_v = view(DA, ntuple(k -> k == dim ? (1:(n - 1)) : axes(A, k), Nd)..., dim)
    DA2_v = view(DA, ntuple(k -> k == dim ? (2:n) : axes(A, k), Nd)..., dim)
    if add
        @. A_v += DA1_v - DA2_v
    else
        @. A_v = DA1_v - DA2_v
    end
    A
end

function D!(DA::AbstractArray{T, N}, A::AbstractArray{T, Nd}) where {N, Nd, T <: Real}
    @assert Nd > 0
    @assert N == Nd + 1
    @assert size(DA)[1:Nd] == size(A)

    # We assume DA was initialized with zeros
    for i in 1:Nd
        finite_difference_forward!(DA, A, i)
        finite_difference_forward!(DA, A, i)
    end
    DA
end

function Dᵀ!(A::AbstractArray{T, Nd}, DA::AbstractArray{T, N}) where {N, Nd, T <: Real}
    @assert Nd > 0
    @assert N == Nd + 1
    @assert size(DA)[1:Nd] == size(A)

    finite_difference_backward!(A, DA, 1)
    for i in 2:Nd
        finite_difference_backward!(A, DA, i, true)
    end

    for i in 1:Nd
        A_v = view(A, ntuple(k -> k == i ? 1 : axes(A, k), Nd)...)
        DA_v = view(DA, ntuple(k -> k == i ? 1 : axes(A, k), Nd)..., i)
        @. A_v = -DA_v
    end
    A
end

struct TVProx <: AbstractProximalOperator
    λ::Float64
    n_iter::Int
    tol::Union{Nothing, Float64}
    isotropic::Bool
    function TVProx(λ::Real, n_iter::Integer = 50; tol::Union{Nothing, Real} = nothing,
            isotropic = true)
        new(λ, n_iter, tol, isotropic)
    end
end

function init(prox::TVProx, x::AbstractArray)
    p = similar(x, (size(x)..., ndims(x)))
    rule = prox.isotropic ? Fista(0.25) : Fista(2.5)
    opt = Optimisers.setup(rule, p)
    ∂p = zero(p)
    y = similar(x)
    (opt, p, ∂p, y)
end

function normalize_if_greater_than_one!(p::AbstractArray, isotropic::Bool)
    if isotropic
        n = norm(p)
        if n > 1
            p ./= n
        end
    else
        @. p = clamp(p, -1, 1)
    end
    p
end

function apply!(prox::TVProx, (opt, p, ∂p, y), y0::AbstractArray{T}) where {T <: Real}
    λ = prox.isotropic ? T(prox.λ) : T(prox.λ/100)
    p .= 0
    y_tmp = isnothing(prox.tol) ? nothing : zero(y)
    opt.state[2] .= 0
    opt.state[3] .= 0
    opt.state = (T(1), opt.state[2:3]...)
    for i in 1:prox.n_iter
        Dᵀ!(y, p)
        @. y = λ*y - y0
        if !isnothing(prox.tol)
            y_tmp .-= y
            max_diff = mapreduce(abs, max, y_tmp)
            if max_diff < prox.tol
                break
            else
                y_tmp .= y
            end
        end
        D!(∂p, y)
        Optimisers.update!(opt, p, ∂p)
        p_slices = eachslice(p, dims = ndims(p))
        foreach(x -> normalize_if_greater_than_one!(x, prox.isotropic), p_slices)
    end
    Dᵀ!(y, p)
    @. y0 = y0 - λ*y
    y0
end

function TV_denoise!(y::AbstractArray, λ::Real, n_iter::Integer = 50;
        tol = nothing, isotropic = true)
    tv = TVProx(λ, n_iter; tol, isotropic)
    tv_state = init(tv, y)
    apply!(tv, tv_state, y)
end
