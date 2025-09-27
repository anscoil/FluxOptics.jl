function finite_difference_forward!(DA::AbstractArray{T},
                                    A::AbstractArray{T, Nd},
                                    dim::Integer) where {Nd, T <: Real}
    @assert dim in 1:Nd
    n = size(A, dim)
    DA_v = view(DA, ntuple(k -> k == dim ? (1:(n - 1)) : axes(A, k), Nd)..., dim)
    A2_v = view(A, ntuple(k -> k == dim ? (2:n) : axes(A, k), Nd)...)
    A1_v = view(A, ntuple(k -> k == dim ? (1:(n - 1)) : axes(A, k), Nd)...)
    @. DA_v = A2_v - A1_v
    DA
end

function finite_difference_backward!(A::AbstractArray{T, Nd},
                                     DA::AbstractArray{T},
                                     dim::Integer,
                                     add = false) where {Nd, T <: Real}
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

"""
    TVProx(λ, n_iter=50; tol=nothing, isotropic=true, rule=Fista(...))

Total Variation (TV) regularization proximal operator.

Apply total variation denoising to promote piecewise smooth solutions in n dimensions.

This works on arrays of any dimensionality: 1D signals, 2D surfaces, 3D volumes,
or higher-dimensional data. Particularly useful for optical applications where
smooth profiles are desired while preserving sharp edges.

# Arguments
- `λ`: Regularization strength (larger values → smoother results)
- `n_iter=50`: Maximum number of internal iterations
- `tol=nothing`: Convergence tolerance (if provided)
- `isotropic=true`: Use isotropic TV (L₂ norm of gradient) vs anisotropic (L₁)
- `rule`: Optimization rule for internal TV solver

# Examples
```jldoctest
julia> tv_smooth = TVProx(0.1, 100; isotropic=false);  # Light anisotropic smoothing

julia> tv_strong = TVProx(1.2, 100);  # Strong isotropic TV

julia> noisy_surface = 1 .+ 0.1*randn(32, 32);

julia> prox_state = ProximalOperators.init(tv_strong, noisy_surface);

julia> strongly_smoothed_surface = ProximalOperators.apply!(tv_strong, prox_state, copy(noisy_surface));
```

See also: [`TV_denoise!`](@ref), [`ProxRule`](@ref)
"""
struct TVProx <: AbstractProximalOperator
    λ::Float64
    n_iter::Int
    tol::Union{Nothing, Float64}
    isotropic::Bool
    rule::AbstractRule

    function TVProx(λ::Real,
                    n_iter::Integer = 50;
                    tol::Union{Nothing, Real} = nothing,
                    isotropic = true,
                    rule = isotropic ? Fista(0.25) : Fista(2.5))
        new(λ, n_iter, tol, isotropic, rule)
    end
end

function init(prox::TVProx, x::AbstractArray)
    p = similar(x, (size(x)..., ndims(x)))
    opt = Optimisers.setup(prox.rule, p)
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
    opt.state = Optimisers.init(opt.rule, p)
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

"""
    TV_denoise!(x, λ, n_iter=50; tol=nothing, isotropic=true, rule=Fista(...))

Apply total variation denoising directly to an n-dimensional array.

Convenience function that applies TV regularization without setting up the full
proximal operator infrastructure. Modifies the input array in-place.

# Arguments
- `x`: Input array to denoise (modified in-place)
- `λ`: Regularization strength
- `n_iter=50`: Maximum iterations
- `tol=nothing`: Convergence tolerance
- `isotropic=true`: Isotropic vs anisotropic TV
- `rule`: Optimization rule for internal TV solver

# Returns
The modified input array (for chaining).

# Examples
```jldoctest
julia> noisy = randn(64, 64) + 5 * sin.(0.1 * (1:64)) * sin.(0.1 * (1:64)');

julia> original_var = var(noisy);

julia> TV_denoise!(noisy, 0.1, 100);

julia> denoised_var = var(noisy);

julia> denoised_var < original_var  # Reduced variation
true

julia> # Can chain operations:
julia> result = TV_denoise!(copy(noisy), 0.05) |> x -> clamp.(x, 0, 1);
```

See also: [`TVProx`](@ref), [`ClampProx`](@ref), [`PositiveProx`](@ref)
"""
function TV_denoise!(y::AbstractArray,
                     λ::Real,
                     n_iter::Integer = 50;
                     tol = nothing,
                     isotropic = true,
                     rule = isotropic ? Fista(0.25) : Fista(2.5))
    tv = TVProx(λ, n_iter; tol, isotropic, rule)
    tv_state = init(tv, y)
    apply!(tv, tv_state, y)
end
