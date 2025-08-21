module Plotting

using Makie
using Makie.Colors
using Makie.ColorSchemes
using LaTeXStrings
using ..Fields

export plot_fields, plot_fields_slider

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

function plot_fields(u_vec, fs::Union{Function, Tuple};
        colormap = :viridis, ratio = 1, max_width = 1024, width = nothing, height = nothing,
        show_colorbars = false, show_labels = false)
    n_lines = length(u_vec)
    @assert n_lines > 0
    n_fields_per_col = length(first(u_vec))
    @assert n_fields_per_col > 0
    fs = isa(fs, Function) ? (fs,) : fs
    ls = length(fs)

    cmaps = isa(colormap, Symbol) ? fill(colormap, ls) : colormap

    if length(cmaps) != ls
        error("Number of colormaps must match number of functions")
    end

    nx, ny = size(first(first(u_vec)))
    fig_width,
    fig_height = get_fig_size(
        ratio*n_fields_per_col*nx*ls, ratio*ny, max_width, width, height)

    width_offset = show_colorbars ? ls*(25 + n_fields_per_col*20) : 0
    height_offset = show_labels ? 50 : 0
    if show_colorbars
        height_offset += 40 * (n_lines-1)
    end
    fig = Figure(size = (fig_width + width_offset, n_lines*fig_height + height_offset))

    if show_labels
        for (j, f) in enumerate(fs)
            fig[1, j] = Label(fig, string(f), halign = :center)
        end
        line_start = 1
    else
        line_start = 0
    end

    for (i, u_fields) in enumerate(u_vec)
        for (j, (f, cmap)) in enumerate(zip(fs, cmaps))
            subgrid = fig[i + line_start, j] = GridLayout()
            colsize!(fig.layout, j, fig_width / ls)
            for (k, u) in enumerate(u_fields)
                cell = subgrid[1, k] = GridLayout()
                ax = Axis(cell[1, 1])
                hidedecorations!(ax)
                ax.aspect = DataAspect()
                hm, is_complex, factor = fill_heatmap!(ax, f, u, cmap)
                if show_colorbars && !is_complex
                    Colorbar(cell[1, 2], hm; width = 10,
                        height = fig_height-n_fields_per_col*20, tickformat = "{:.1f}")
                    if !iszero(factor)
                        lbl = Label(cell[1, 2, Top()], LaTeXString("\\times 10^{$factor}"))
                        lbl.padding[] = (0, 0, 2, 0)
                    end
                end
            end
        end
    end

    fig
end

function plot_fields(
        u_vec::Union{AbstractVector{U}, Tuple{Vararg{U}}},
        fs::Union{Function, Tuple};
        colormap = :viridis, ratio = 1, max_width = 1024,
        width = nothing, height = nothing,
        show_colorbars = false, show_labels = false) where {
        U <: Union{ScalarField, AbstractArray{<:Number}}}
    plot_fields(map(u -> (get_data(u),), u_vec), fs; colormap = colormap,
        ratio = ratio, max_width = max_width,
        width = width, height = height, show_colorbars = show_colorbars, show_labels = show_labels)
end

function plot_fields(u::Union{ScalarField, AbstractArray{T}}, fs::Union{Function, Tuple};
        colormap = :viridis, ratio = 1, max_width = 1024,
        width = nothing, height = nothing, show_colorbars = false, show_labels = false
) where {T <: Number}
    plot_fields(((get_data(u),),), fs; colormap = colormap, ratio = ratio,
        max_width = max_width, width = width, height = height,
        show_colorbars = show_colorbars, show_labels = show_labels)
end

function plot_fields_slider(u_vec, fs::Union{Function, Tuple};
        colormap = :viridis, ratio = 1, max_width = Inf, width = nothing, height = nothing)
    n_lines = length(u_vec)
    @assert n_lines > 0
    n_fields_per_col = length(first(u_vec))
    @assert n_fields_per_col > 0

    fs = isa(fs, Function) ? (fs,) : fs
    ls = length(fs)

    cmaps = isa(colormap, Symbol) ? fill(colormap, ls) : colormap
    @assert length(cmaps) == ls

    nx, ny = size(first(first(u_vec)))
    fig_width,
    fig_height = get_fig_size(
        ratio*n_fields_per_col*nx*ls, ratio*ny, max_width, width, height)

    fig = Figure(size = (fig_width, fig_height + 80))  # + extra for slider

    for (j, f) in enumerate(fs)
        fig[1, j] = Label(fig, string(f), halign = :center)
        colsize!(fig.layout, j, fig_width / ls - 30)
    end

    heatmaps = []

    for (j, (f, cmap)) in enumerate(zip(fs, cmaps))
        subgrid = fig[2, j] = GridLayout()
        for k in 1:n_fields_per_col
            ax = Axis(subgrid[1, k])
            hidedecorations!(ax)
            ax.aspect = DataAspect()
            push!(heatmaps, (ax, f, cmap, k))
        end
    end

    sl = Slider(fig[3, 1:ls], range = 1:n_lines, startvalue = 1)

    u_vec = collect(u_vec)
    on(sl.value) do i
        u_fields = collect(u_vec[i])
        for (ax, f, cmap, k) in heatmaps
            u = u_fields[k]
            fill_heatmap!(ax, f, u, cmap)
        end
    end

    notify(sl.value)

    fig
end

function plot_fields_slider(
        u_vec::Union{AbstractVector{U}, Tuple{Vararg{U}}},
        fs::Union{Function, Tuple};
        colormap = :viridis, ratio = 1, max_width = 2048,
        width = nothing, height = nothing) where {T <: Number, U <: AbstractArray{T}}
    plot_fields_slider(map(u -> (get_data(u),), u_vec), fs; colormap = colormap,
        ratio = ratio, max_width = max_width,
        width = width, height = height)
end

end
