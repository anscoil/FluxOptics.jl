module Plotting

using Makie
using Makie.Colors

export complex_to_rgb, plot_fields

function complex_to_rgb(
        A::AbstractArray{Complex{T}}; colormap = :dark, rmax = nothing) where {T}
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

function get_fig_size(nx, ny, max_width, width = nothing, height = nothing)
    if isnothing(width)
        if isnothing(height)
            w = min(max_width, nx)
            return (w, w*ny/nx)
        else
            w = nx*height/ny
            if w > max_width
                error("Exceeding max_width for height = $(height)")
            else
                return (w, height)
            end
        end
    else
        if isnothing(height)
            if width > max_width
                error("Width exceeding max_width: increase max_width or lower width")
            end
            return (width, width*ny/nx)
        else
            error("Overconstrained plot: specify only one of width or height")
        end
    end
end

function plot_fields(
        u_vec::Union{AbstractVector{<:U}, Tuple{Vararg{<:U}}}, fs::Union{Function, Tuple};
        colormap = :viridis, ratio = 1, max_width = 1024,
        width = nothing, height = nothing) where {T, U <: AbstractArray{T, 2}}
    @assert length(u_vec) > 0
    fs = isa(fs, Function) ? (fs,) : fs
    ls = length(fs)
    n_fields = length(u_vec)
    cmaps = isa(colormap, Symbol) ? fill(colormap, ls) : colormap

    if length(cmaps) != length(fs)
        error("Number of colormaps must match number of functions")
    end

    nx, ny = size(u_vec[1])
    fig_width, fig_height = get_fig_size(ratio*nx*ls, ratio*ny, max_width, width, height)

    fig = Figure(size = (fig_width, n_fields*fig_height))
    for (i, u) in enumerate(u_vec)
        for (j, (f, cmap)) in enumerate(zip(fs, cmaps))
            ax = Axis(fig[i, j])
            if i == 1
                ax.title = string(f)
            end
            hidedecorations!(ax)
            colsize!(fig.layout, j, Aspect(1, nx/ny))
            img = f.(u)
            if eltype(img) <: Complex
                if colormap == :viridis
                    cmap = :dark
                end
                heatmap!(ax, complex_to_rgb(img; colormap = cmap))
            else
                heatmap!(ax, img; colormap = cmap)
            end
        end
    end

    fig
end

function plot_fields(u::AbstractArray{T, 2}, fs::Union{Function, Tuple};
        colormap = :viridis, ratio = 1, max_width = 1024,
        width = nothing, height = nothing) where {T}
    plot_fields((u,), fs; colormap = colormap, ratio = ratio, max_width = max_width,
        width = width, height = height)
end

end
