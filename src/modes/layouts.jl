const Layout2D = Vector{Tuple{T, T}} where {T <: Real}

function Base.map(t::AbstractAffineMap, layout::Layout2D)
    map(x -> Tuple(t(x...)), layout)
end
    
function triangle_layout(np::Integer, px, py)
    layout = Tuple{Float64, Float64}[]
    for i in np:-1:1
        for j in 1:i
            xp = ((i-1)-(np-1)/2)*px
            yp = ((j-1)-(np-1)/2)*py
            push!(layout, (xp, yp))
        end
    end
    layout
end

function triangle_layout(np::Integer, px, py, t::AbstractAffineMap)
    map(t, triangle_layout(np, px, py))
end

function generate_mode_stack(layout::Layout2D, nx, ny, dx, dy, m::Mode;
        t::AbstractAffineMap = Id2D(), normalize = true)
    n_modes = length(layout)
    xv, yv = spatial_vectors(nx, ny, dx, dy)
    modes = zeros(eltype(m), (nx, ny, n_modes))
    for (k, pos) in enumerate(layout)
        mode = m(@view(modes[:, :, k]), xv, yv, t ∘ Shift2D(pos...))
        if normalize
            mode ./= norm(mode)
        end
    end
    modes
end

function generate_mode_stack(
        layout::Layout2D, nx, ny, dx, dy, m_v::AbstractVector{<:Mode};
        t::AbstractAffineMap = Id2D(), normalize = true)
    @assert length(layout) == length(m_v)
    n_modes = length(layout)
    xv, yv = spatial_vectors(nx, ny, dx, dy)
    modes = zeros(reduce(promote_type, eltype.(m_v)), (nx, ny, n_modes))
    for (k, (pos, m)) in enumerate(zip(layout, m_v))
        mode = m(@view(modes[:, :, k]), xv, yv, t ∘ Shift2D(pos...))
        if normalize
            mode ./= norm(mode)
        end
    end
    modes
end

function generate_mode_stack(nx, ny, dx, dy, m_v::AbstractVector{<:Mode};
        t::AbstractAffineMap = Id2D(), normalize = true)
    n = length(m_v)
    layout = [(0, 0) for _ in 1:n]
    generate_mode_stack(layout, nx, ny, dx, dy, m_v; t = t, normalize = normalize)
end
