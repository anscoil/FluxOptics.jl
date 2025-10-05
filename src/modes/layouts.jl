abstract type AbstractLayout2D end

"""
    PointLayout(n, p=(0,0), t=Id2D())

Create a layout of n identical points.

# Arguments
- `n`: Number of points
- `p=(0,0)`: Position of the point
- `t=Id2D()`: Coordinate transformation

# Returns
`PointLayout` that can be iterated to get point positions.

# Examples
```jldoctest
julia> layout = Modes.PointLayout(3, (100.0, 50.0));

julia> positions = collect(layout);

julia> length(positions)
3

julia> all(pos == (100.0, 50.0) for pos in positions)
true
```
"""
struct PointLayout <: AbstractLayout2D
    n::Int
    p::Tuple{Float64, Float64}
    t::AbstractAffineMap

    function PointLayout(n, p = (0, 0), t = Id2D())
        new(n, p, t)
    end
end

Base.length(l::PointLayout) = l.n

function Base.iterate(l::PointLayout, state = 1)
    i = state
    if i > length(l)
        return nothing
    end
    point = Tuple(l.t(l.p...))
    next_state = i+1
    point, next_state
end

"""
    GridLayout(nx, ny, px, py, t=Id2D())

Create a regular rectangular grid layout.

# Arguments
- `nx, ny`: Number of points in x and y directions
- `px, py`: Spacing between points in x and y
- `t=Id2D()`: Coordinate transformation

# Returns
`GridLayout` that can be iterated to get point positions.

# Examples
```jldoctest
julia> layout = Modes.GridLayout(2, 3, 100.0, 50.0);

julia> positions = collect(layout);

julia> length(positions)  # 2×3 = 6 points
6

julia> positions[1]  # First point (centered grid)
(-50.0, -50.0)

julia> positions[end]  # Last point
(50.0, 50.0)
```
"""
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

Base.length(l::GridLayout) = l.nx * l.ny

function Base.iterate(l::GridLayout, state = (1, 1))
    i, j = state
    if i > l.nx
        return nothing
    end
    xp = ((i-1) - (l.nx-1)/2) * l.px
    yp = ((j-1) - (l.ny-1)/2) * l.py
    point = Tuple(l.t(xp, yp))
    if j < l.ny
        next_state = (i, j+1)
    else
        next_state = (i+1, 1)
    end
    point, next_state
end

"""
    TriangleLayout(np, px, py, t=Id2D())

Create a triangular arrangement of points.

# Arguments
- `np`: Number of points along triangle edge
- `px, py`: Spacing between points in x and y
- `t=Id2D()`: Coordinate transformation

# Returns
`TriangleLayout` with np(np+1)/2 total points.

# Examples
```jldoctest
julia> layout = Modes.TriangleLayout(3, 100.0, 100.0);

julia> length(layout)  # 3×4/2 = 6 points
6

julia> layout = Modes.TriangleLayout(4, 50.0, 50.0);

julia> length(layout)  # 4×5/2 = 10 points
10
```
"""
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
    point = Tuple(l.t(xp, yp))
    if j < i
        next_state = (i, j+1)
    else
        next_state = (i-1, 1)
    end
    point, next_state
end

"""
    CustomLayout(positions, t=Id2D())

Create a layout from custom list of positions.

# Arguments
- `positions`: Vector of (x,y) tuples specifying point positions
- `t=Id2D()`: Coordinate transformation applied to all positions

# Returns
`CustomLayout` that can be iterated to get transformed positions.

# Examples
```jldoctest
julia> positions = [(0.0, 0.0), (100.0, 0.0), (50.0, 86.6)];

julia> layout = Modes.CustomLayout(positions);

julia> collect(layout) == positions
true

julia> layout_rot = Modes.CustomLayout(positions, Rot2D(π/2));

julia> rotated = collect(layout_rot);

julia> all(isapprox.(rotated[2], (0.0, 100.0), atol=1e-12))  # Second point rotated 90°
true
```
"""
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
    point = Tuple(l.t(l.l[i]...))
    next_state = i+1
    point, next_state
end

"""
    generate_mode_stack(layout, nx, ny, dx, dy, mode; t=Id2D(), normalize=true)
    generate_mode_stack(layout, nx, ny, dx, dy, mode_vector; t=Id2D(), normalize=true)
    generate_mode_stack(nx, ny, dx, dy, mode_vector; t=Id2D(), normalize=true)

Generate arrays of optical modes at specified positions.

The first form replicates the same mode at each layout position. The second form
uses different modes from mode_vector at each position. The third form places
modes at the origin (single position each).

# Arguments
- `layout`: Spatial layout determining mode positions
- `nx, ny`: Grid size for each mode field
- `dx, dy`: Pixel size
- `mode`: Single mode to replicate at all positions
- `mode_vector`: Vector of modes (one per position)
- `t=Id2D()`: Additional transformation applied to all modes
- `normalize=true`: Normalize each mode to unit power

# Returns
3D complex array of size (nx, ny, n_modes) containing mode fields.

# Examples
```jldoctest
julia> layout = Modes.GridLayout(2, 2, 50.0, 50.0);

julia> gaussian = Gaussian(10.0);

julia> modes = generate_mode_stack(layout, 64, 64, 2.0, 2.0, gaussian);

julia> size(modes)
(64, 64, 4)

julia> sum(abs2, modes[:, :, 1]) * 2.0 * 2.0  # Check first mode normalization
1.0000000000000007

julia> hg_modes = [HermiteGaussian(25.0, m, n) for m in 0:1 for n in 0:1];

julia> modes_hg = generate_mode_stack(layout, 64, 64, 2.0, 2.0, hg_modes);

julia> size(modes_hg)
(64, 64, 4)

julia> lg_modes = [LaguerreGaussian(20.0, 0, l) for l in 0:2];

julia> modes_lg = generate_mode_stack(64, 64, 2.0, 2.0, lg_modes);

julia> size(modes_lg)  # No layout specified, modes at origin
(64, 64, 3)
```
"""
function generate_mode_stack(layout::AbstractLayout2D,
                             nx,
                             ny,
                             dx,
                             dy,
                             m::Mode{Nd, T};
                             t::AbstractAffineMap = Id2D(),
                             normalize = true) where {Nd, T}
    n_modes = length(layout)
    xv, yv = spatial_vectors(nx, ny, dx, dy)
    modes = zeros(Complex{T}, (nx, ny, n_modes))
    for (k, pos) in enumerate(layout)
        mode = m(@view(modes[:, :, k]), xv, yv, Shift2D(pos...) ∘ t)
        if normalize
            mode ./= (norm(mode) * sqrt(dx*dy))
        end
    end
    modes
end

function generate_mode_stack(layout::AbstractLayout2D,
                             nx,
                             ny,
                             dx,
                             dy,
                             m_v::AbstractVector{<:Mode};
                             t::AbstractAffineMap = Id2D(),
                             normalize = true)
    @assert length(layout) == length(m_v)
    n_modes = length(layout)
    xv, yv = spatial_vectors(nx, ny, dx, dy)
    T = reduce(promote_type, eltype.(m_v))
    modes = zeros(T, (nx, ny, n_modes))
    for (k, (pos, m)) in enumerate(zip(layout, m_v))
        mode = m(@view(modes[:, :, k]), xv, yv, Shift2D(pos...) ∘ t)
        if normalize
            mode ./= (norm(mode) * sqrt(dx*dy))
        end
    end
    modes
end

function generate_mode_stack(nx,
                             ny,
                             dx,
                             dy,
                             m_v::AbstractVector{<:Mode};
                             t::AbstractAffineMap = Id2D(),
                             normalize = true)
    n = length(m_v)
    layout = PointLayout(n)
    generate_mode_stack(layout, nx, ny, dx, dy, m_v; t, normalize)
end
