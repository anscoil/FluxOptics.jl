abstract type AbstractLayout2D end

struct PointLayout <: AbstractLayout2D
    n::Int
    p::Tuple{Float64, Float64}
    t::AbstractAffineMap

    function PointLayout(n, p = (0, 0), t = Id2D())
        new(n, p, t)
    end
end

Base.eltype(::Type{PointLayout}) = Tuple{Float64, Float64}
Base.length(l::PointLayout) = l.n

function Base.iterate(l::PointLayout, state = 1)
    i = state
    if i > length(l)
        return nothing
    end
    point = l.t(l.p...)
    next_state = i+1
    point, next_state
end

struct GridLayout <: AbstractLayout2D
    nx::Int
    ny::Int
    px::Float64
    py::Float64
    t::AbstractAffineMap

    function GridLayout(nx, ny, px, py, t = Id2D())
        new(nx, ny, px, py, t)
    end
end

Base.eltype(::Type{GridLayout}) = Tuple{Float64, Float64}
Base.length(l::GridLayout) = l.nx * l.ny

function Base.iterate(l::GridLayout, state = (1, 1))
    i, j = state
    if i > l.nx
        return nothing
    end
    xp = ((i-1) - (l.nx-1)/2) * l.px
    yp = ((j-1) - (l.ny-1)/2) * l.py
    point = l.t(xp, yp)
    if j < l.ny
        next_state = (i, j+1)
    else
        next_state = (i+1, 1)
    end
    point, next_state
end

struct TriangleLayout <: AbstractLayout2D
    np::Int
    px::Float64
    py::Float64
    t::AbstractAffineMap

    function TriangleLayout(np, px, py, t = Id2D())
        new(np, px, py, t)
    end
end

Base.eltype(::Type{TriangleLayout}) = Tuple{Float64, Float64}
Base.length(l::TriangleLayout) = div(l.np*(l.np + 1), 2)

function Base.iterate(l::TriangleLayout, state = (l.np, 1))
    i, j = state
    if i < 1
        return nothing
    end
    xp = ((i-1) - (l.np-1)/2) * l.px
    yp = ((j-1) - (l.np-1)/2) * l.py
    point = l.t(xp, yp)
    if j < i
        next_state = (i, j+1)
    else
        next_state = (i-1, 1)
    end
    point, next_state
end

struct CustomLayout <: AbstractLayout2D
    l::Vector{Tuple{Float64, Float64}}
    t::AbstractAffineMap

    function CustomLayout(l, t = Id2D())
        new(l, t)
    end
end

Base.eltype(::Type{CustomLayout}) = Tuple{Float64, Float64}
Base.length(l::CustomLayout) = length(l.l)

function Base.iterate(l::CustomLayout, state = 1)
    i = state
    if i > length(l)
        return nothing
    end
    point = l.t(l.l[i]...)
    next_state = i+1
    point, next_state
end

function generate_mode_stack(layout::AbstractLayout2D, nx, ny, dx, dy, m::Mode;
        t::AbstractAffineMap = Id2D(), normalize = true)
    n_modes = length(layout)
    xv, yv = spatial_vectors(nx, ny, dx, dy)
    modes = zeros(eltype(m), (nx, ny, n_modes))
    for (k, pos) in enumerate(layout)
        mode = m(@view(modes[:, :, k]), xv, yv, Shift2D(pos...) ∘ t)
        if normalize
            mode ./= norm(mode)
        end
    end
    modes
end

function generate_mode_stack(
        layout::AbstractLayout2D, nx, ny, dx, dy, m_v::AbstractVector{<:Mode};
        t::AbstractAffineMap = Id2D(), normalize = true)
    @assert length(layout) == length(m_v)
    n_modes = length(layout)
    xv, yv = spatial_vectors(nx, ny, dx, dy)
    modes = zeros(reduce(promote_type, eltype.(m_v)), (nx, ny, n_modes))
    for (k, (pos, m)) in enumerate(zip(layout, m_v))
        mode = m(@view(modes[:, :, k]), xv, yv, Shift2D(pos...) ∘ t)
        if normalize
            mode ./= norm(mode)
        end
    end
    modes
end

function generate_mode_stack(nx, ny, dx, dy, m_v::AbstractVector{<:Mode};
        t::AbstractAffineMap = Id2D(), normalize = true)
    n = length(m_v)
    layout = PointLayout(n)
    generate_mode_stack(layout, nx, ny, dx, dy, m_v; t = t, normalize = normalize)
end
