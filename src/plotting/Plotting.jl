module Plotting

using Makie
using Makie.Colors
using Makie.ColorSchemes
using LaTeXStrings
using ..Fields
using ..OpticalComponents

export visualize, visualize_slider

function complex_to_rgb(A::AbstractArray{Complex{T}};
                        colormap = :dark,
                        rmax = nothing) where {T}
    A_abs = abs.(A)
    absmax = isnothing(rmax) ? maximum(A_abs) : rmax

    hue = ((angle.(A) .+ 2π) .% 2π) ./ 2π * 360

    if colormap == :light
        sat = clamp.(A_abs ./ absmax, 0.0, 1.0)
        val = ones(size(A))
    elseif colormap == :dark
        sat = ones(size(A))
        val = clamp.(A_abs ./ absmax, 0.0, 1.0)
    else
        error("No such colormap $(colormap) for complex arrays")
    end

    rgb_colors = RGBf.(Colors.HSV.(hue, sat, val))
    rgb_colors
end

function valid_colormap(name::Symbol)
    name in keys(ColorSchemes.colorschemes) || error("Invalid colormap: $name")
    name
end

function fill_heatmap!(ax, f, u, cmap)
    img = f(u)
    if ndims(img) != 2
        error("Function $(f) failed to reduce the input to a 2-dimensional array")
    end
    if eltype(img) <: Complex
        if cmap == :viridis
            cmap = :dark
        end
        hm = heatmap!(ax, complex_to_rgb(img; colormap = cmap)), true, nothing
    else
        maxval = maximum(abs.(img))
        factor = iszero(maxval) ? 0 : Int(floor(log10(maxval)))
        if !iszero(factor)
            img ./= 10.0^factor
        end
        hm = heatmap!(ax, img; colormap = valid_colormap(cmap)), false, factor
    end
    hm
end

function parse_args(args, n_cols, n_fields_per_col)
    if !isa(args, Tuple)
        ntuple(_ -> ntuple(_ -> args, n_fields_per_col), n_cols)
    else
        @assert length(args) == n_cols
        map(x -> if !isa(x, Tuple)
                ntuple(_ -> x, n_fields_per_col)
            else
                @assert length(x) == n_fields_per_col
                x
            end, args)
    end
end

Plottable = Union{ScalarField, AbstractOpticalComponent,
                  AbstractArray{T, 2} where {T <: Number}}

"""
    visualize(u, fs; colormap=:viridis, height=200, show_colorbars=false)
    visualize(u_vec, fs; colormap=:viridis, height=200, show_colorbars=false)

Display optical fields or components with specified visualization functions.

Creates a static heatmap figure showing one or more fields with one or more
visualization functions (intensity, phase, etc.). All views are displayed
simultaneously in a grid layout. Works with CairoMakie or GLMakie.

# Arguments
- `u`: Single plottable object (ScalarField, AbstractOpticalComponent, or 2D array)
- `u_vec`: Vector or tuple of plottable objects (displayed as rows)
- `fs`: Visualization function or tuple of functions (displayed as columns)
- `colormap`: Colormap name(s) - single symbol or tuple matching `fs`
- `height`: Height in pixels per heatmap (default: 200)
- `show_colorbars`: Display colorbars for real-valued plots (default: false)

# Plottable Types
- `ScalarField`: Optical fields
- `AbstractOpticalComponent`: Phase masks, DOEs, etc. with 2D data
- `AbstractArray{<:Number, 2}`: Raw 2D arrays

# Visualization Functions
Common functions:
- `intensity`: Total intensity |u|²
- `phase`: Phase angle in radians
- `real`, `imag`: Real/imaginary parts
- `abs`: Magnitude
- `identity`: Complex field with HSV colormap

# Examples

**Single field, single view:**
```julia
u = ScalarField(gaussian_data, (2.0, 2.0), 1.064)
fig = visualize(u, intensity)
```

**Multiple representations:**
```julia
fig = visualize(u, (intensity, phase, real, imag))
```

**Multiple fields (comparison):**
```julia
fig = visualize([target, optimized], intensity)
```

**Custom colormaps:**
```julia
fig = visualize(u, (intensity, phase); 
                colormap=(:viridis, :twilight))
```

**With colorbars:**
```julia
fig = visualize(u, intensity; show_colorbars=true, height=300)
```

**Visualize component:**
```julia
phase_mask = Phase(u, (x, y) -> x^2; trainable=true)
fig = visualize(phase_mask, identity)
```

# Notes
- For large field stacks, prefer `visualize_slider` to avoid displaying all at once
- Complex-valued results automatically use HSV colormap (`:dark` or `:light`)
- Width computed automatically to maintain aspect ratio
- Compatible with CairoMakie (static) and GLMakie (interactive)

See also: [`visualize_slider`](@ref), [`intensity`](@ref), [`phase`](@ref)
"""
function visualize(u_vec,
                   fs::Union{Function, Tuple};
                   colormap = :viridis,
                   height = 200,
                   show_colorbars = false)
    n_lines = length(u_vec)
    @assert n_lines > 0
    n_fields_per_col = length(first(u_vec))
    n_cols = isa(fs, Tuple) ? length(fs) : 1
    @assert n_fields_per_col > 0
    fs = parse_args(fs, n_cols, n_fields_per_col)

    cmaps = parse_args(colormap, n_cols, n_fields_per_col)

    fig = Figure()

    for (i, u_fields) in enumerate(u_vec)
        for (j, (f, cmap)) in enumerate(zip(fs, cmaps))
            subgrid = fig[i, j] = GridLayout()
            for (k, u) in enumerate(u_fields)
                img = collect(u)
                nx, ny = size(img)
                width = nx*height/ny
                cell = subgrid[1, k] = GridLayout()
                ax = Axis(cell[1, 1], width = width, height = height)
                hidedecorations!(ax)
                hm, is_complex, factor = fill_heatmap!(ax, f[k], img, cmap[k])
                if show_colorbars && !is_complex
                    Colorbar(cell[1, 2],
                             hm;
                             width = 10,
                             height = height,
                             tickformat = "{:.1f}")
                    if !iszero(factor)
                        lbl = Label(cell[1, 2, Top()], LaTeXString("\\times 10^{$factor}"))
                        lbl.padding[] = (0, 0, 2, 0)
                    end
                end
            end
        end
    end
    resize_to_layout!(fig)
    fig
end

function visualize(u_vec::Union{AbstractVector{U}, Tuple{Vararg{U}}},
                   fs::Union{Function, Tuple};
                   colormap = :viridis,
                   height = 200,
                   show_colorbars = false) where {U <: Plottable}
    visualize(map(u -> (collect(u),), u_vec), fs; colormap, height, show_colorbars)
end

function visualize(u::Plottable,
                   fs::Union{Function, Tuple};
                   colormap = :viridis,
                   ratio = 1,
                   max_width = 1024,
                   width = nothing,
                   height = 200,
                   show_colorbars = false)
    visualize(((collect(u),),), fs; colormap, height, show_colorbars)
end

"""
    visualize_slider(u_vec, fs; colormap=:viridis, height=400)

Display field sequence with interactive slider for navigation.

Creates an interactive figure with a horizontal slider to navigate through a
sequence of fields. All visualization functions update in real-time when
moving the slider. **Requires GLMakie** (interactive backend).

This is preferred over `visualize` for large field sequences (propagation steps,
optimization iterations, etc.) as it avoids displaying all fields simultaneously.

# Arguments
- `u_vec`: Vector or tuple of plottable objects (sequence to navigate)
- `fs`: Visualization function or tuple of functions
- `colormap`: Colormap name(s) - single symbol or tuple matching `fs`
- `height`: Height in pixels per heatmap (default: 400)

# Requirements
- **Must use GLMakie** (interactive backend with slider support)
- CairoMakie and WGLMakie do not support interactive sliders

# Examples

**Propagation sequence:**
```julia
using GLMakie

# Create propagation sequence
distances = range(0, 2000, length=50)
u0 = ScalarField(gaussian_data, (2.0, 2.0), 1.064)
sequence = [propagate(u0, ASProp(u0, z), Forward) for z in distances]

# Interactive slider
fig = visualize_slider(sequence, intensity)
display(fig)
```

**Multiple representations:**
```julia
fig = visualize_slider(sequence, (intensity, phase))
```

**Optimization iterations:**
```julia
# Save field at each iteration
history = ScalarField[]
for iter in 1:100
    # ... optimization step ...
    push!(history, copy(system()))
end

fig = visualize_slider(history, (intensity, phase))
```

**Custom colormaps:**
```julia
fig = visualize_slider(sequence, (intensity, phase);
                       colormap=(:inferno, :twilight),
                       height=500)
```

# Use Cases
- Propagation through optical system
- Optimization convergence visualization
- Wavelength sweeps
- Parameter scans
- Time evolution

# Performance
- Only current frame is rendered (efficient for large sequences)
- Slider provides smooth navigation
- Consider downsampling very large fields if interactive response is slow

# Notes
- Slider range: 1 to length(u_vec)
- All views update synchronously
- **GLMakie required** - will error with CairoMakie

See also: [`visualize`](@ref), [`intensity`](@ref), [`phase`](@ref)
"""
function visualize_slider(u_vec,
                          fs::Union{Function, Tuple};
                          colormap = :viridis,
                          height = 400)
    n_lines = length(u_vec)
    @assert n_lines > 0
    n_fields_per_col = length(first(u_vec))
    n_cols = isa(fs, Tuple) ? length(fs) : 1
    @assert n_fields_per_col > 0
    fs = parse_args(fs, n_cols, n_fields_per_col)

    cmaps = parse_args(colormap, n_cols, n_fields_per_col)

    u_vec = collect(u_vec)
    u_data = map(u_fields -> map(u -> collect(u), collect(u_fields)), u_vec)

    fig = Figure()

    heatmaps = []

    for (j, (f, cmap)) in enumerate(zip(fs, cmaps))
        subgrid = fig[1, j] = GridLayout()
        for k in 1:n_fields_per_col
            nx, ny = size(u_data[j][k])
            width = nx*height/ny
            ax = Axis(subgrid[1, k], width = width, height = height)
            hidedecorations!(ax)
            ax.aspect = DataAspect()
            push!(heatmaps, (ax, f, cmap, k))
        end
    end

    sl = Slider(fig[2, 1:n_cols], range = 1:n_lines, startvalue = 1)

    on(sl.value) do i
        u_fields = collect(u_vec[i])
        for (ax, f, cmap, k) in heatmaps
            u = collect(u_fields[k])
            fill_heatmap!(ax, f[k], u, cmap[k])
        end
    end

    notify(sl.value)
    resize_to_layout!(fig)
    fig
end

function visualize_slider(u_vec::Union{AbstractVector{U}, Tuple{Vararg{U}}},
                          fs::Union{Function, Tuple};
                          colormap = :viridis,
                          height = 400) where {U <: Plottable}
    visualize_slider(map(u -> (collect(u),), u_vec), fs; colormap, height)
end

end
