function ista(p, c)
    function shrink(x::T) where {T}
        s = T(p)
        xc = x - T(c)
        if abs(xc) <= s
            return T(c)
        elseif xc > s
            return x - s
        else
            return x + s
        end
    end
    return shrink
end

"""
    IstaProx(λ, center=0)

Iterative Shrinkage-Thresholding (ISTA) proximal operator for sparse optimization.

Applies soft thresholding to promote sparsity in the parameters. Values close to
`center` are shrunk towards `center`, promoting sparse solutions.

# Arguments
- `λ`: Shrinkage parameter (larger values → more sparsity)
- `center=0`: Center value for thresholding

# Examples
```jldoctest
julia> sparse_prox = IstaProx(0.01, 0.0);  # Shrink towards zero

julia> x = [-0.5, -0.005, 0.0, 0.003, 0.2];

julia> prox_state = ProximalOperators.init(sparse_prox, x);

julia> ProximalOperators.apply!(sparse_prox, prox_state, x);

julia> x  # Small values shrunk to zero
5-element Vector{Float64}:
 -0.49
  0.0
  0.0
  0.0
  0.19
```

See also: [`PointwiseProx`](@ref), [`ProxRule`](@ref), [`Fista`](@ref)
"""
function IstaProx(s::Real, c::Real = 0)
    PointwiseProx(ista(s, c))
end
