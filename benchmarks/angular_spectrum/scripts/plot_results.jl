using JSON, CairoMakie, Statistics

results_dir = joinpath(@__DIR__, "../results")

# Load data
data_fo = JSON.parsefile(joinpath(results_dir, "fluxoptics_bench.json"))
data_pure = JSON.parsefile(joinpath(results_dir, "fluxoptics_pure_bench.json"))
data_wave = JSON.parsefile(joinpath(results_dir, "waveopticspropagation_bench.json"))
data_jax = JSON.parsefile(joinpath(results_dir, "jaxoptics_bench.json"))
data_torch = JSON.parsefile(joinpath(results_dir, "torchoptics_bench.json"))

# Extraction helper
function extract(data, device)
    d = data[device]
    nmodes = [x["n_modes"] for x in d]
    times = [x["time_median_ms"] for x in d]
    return nmodes, times
end

function interpolate_loglog(nm_ref, t_ref, nm_target)
    log_nm_ref = log10.(nm_ref)
    log_t_ref = log10.(t_ref)
    log_nm_target = log10.(nm_target)

    t_interp = similar(nm_target, Float64)

    for (i, log_nm) in enumerate(log_nm_target)
        if log_nm <= log_nm_ref[1]
            t_interp[i] = 10^log_t_ref[1]
        elseif log_nm >= log_nm_ref[end]
            t_interp[i] = 10^log_t_ref[end]
        else
            j = searchsortedfirst(log_nm_ref, log_nm)
            if j == 1
                j = 2
            end

            x0, x1 = log_nm_ref[j-1], log_nm_ref[j]
            y0, y1 = log_t_ref[j-1], log_t_ref[j]

            log_t = y0 + (y1 - y0) * (log_nm - x0) / (x1 - x0)
            t_interp[i] = 10^log_t
        end
    end

    return t_interp
end

function compute_speedup_with_interp(nm_ref, t_ref, nm_test, t_test)
    t_ref_interp = interpolate_loglog(nm_ref, t_ref, nm_test)
    speedup = t_ref_interp ./ t_test
    return nm_test, speedup
end

# Extract data
nm_fo, t_fo = extract(data_fo, "gpu")
nm_pure, t_pure = extract(data_pure, "gpu")
nm_wave, t_wave = extract(data_wave, "gpu")
nm_jax, t_jax = extract(data_jax, "gpu")
nm_torch, t_torch = extract(data_torch, "gpu")

nm_fo_cpu, t_fo_cpu = extract(data_fo, "cpu")
nm_pure_cpu, t_pure_cpu = extract(data_pure, "cpu")
nm_wave_cpu, t_wave_cpu = extract(data_wave, "cpu")
nm_jax_cpu, t_jax_cpu = extract(data_jax, "cpu")
nm_torch_cpu, t_torch_cpu = extract(data_torch, "cpu")

# Compute speedups vs FluxOptics custom (baseline)
nm_speedup_pure, speedup_pure = compute_speedup_with_interp(nm_fo, t_fo, nm_pure, t_pure)
nm_speedup_wave, speedup_wave = compute_speedup_with_interp(nm_fo, t_fo, nm_wave, t_wave)
nm_speedup_jax, speedup_jax = compute_speedup_with_interp(nm_fo, t_fo, nm_jax, t_jax)
nm_speedup_torch,
speedup_torch = compute_speedup_with_interp(nm_fo, t_fo, nm_torch, t_torch)

nm_speedup_pure_cpu,
speedup_pure_cpu = compute_speedup_with_interp(nm_fo_cpu, t_fo_cpu, nm_pure_cpu, t_pure_cpu)
nm_speedup_wave_cpu,
speedup_wave_cpu = compute_speedup_with_interp(nm_fo_cpu, t_fo_cpu, nm_wave_cpu, t_wave_cpu)
nm_speedup_jax_cpu,
speedup_jax_cpu = compute_speedup_with_interp(nm_fo_cpu, t_fo_cpu, nm_jax_cpu, t_jax_cpu)
nm_speedup_torch_cpu,
speedup_torch_cpu = compute_speedup_with_interp(nm_fo_cpu, t_fo_cpu, nm_torch_cpu,
                                                t_torch_cpu)

# ============================================================================
# FIGURE 1: GPU vs CPU Performance
# ============================================================================

fig1 = Figure(size = (1600, 800), fontsize = 18)

# GPU subplot
ax_gpu = Axis(fig1[1, 1],
              xlabel = "Number of modes",
              ylabel = "Time (ms)",
              title = "GPU Bidirectional Propagation Performance",
              xscale = log10,
              yscale = log10)

scatterlines!(ax_gpu, nm_fo, t_fo,
              label = "FluxOptics (custom)",
              color = :blue, linewidth = 3,
              markersize = 12, marker = :circle)

scatterlines!(ax_gpu, nm_pure, t_pure,
              label = "FluxOptics (pure)",
              color = :orange, linewidth = 3,
              markersize = 12, marker = :utriangle)

scatterlines!(ax_gpu, nm_wave, t_wave,
              label = "WaveOpticsPropagation",
              color = :red, linewidth = 3,
              markersize = 12, marker = :rect)

scatterlines!(ax_gpu, nm_jax, t_jax,
              label = "JaxOptics",
              color = :green, linewidth = 3,
              markersize = 12, marker = :diamond)

scatterlines!(ax_gpu, nm_torch, t_torch,
              label = "TorchOptics",
              color = :purple, linewidth = 3,
              markersize = 12, marker = :star5)

axislegend(ax_gpu, position = :rb)

# CPU subplot
ax_cpu = Axis(fig1[1, 2],
              xlabel = "Number of modes",
              ylabel = "Time (ms)",
              title = "CPU Bidirectional Propagation Performance",
              xscale = log10,
              yscale = log10)

scatterlines!(ax_cpu, nm_fo_cpu, t_fo_cpu,
              label = "FluxOptics (custom)",
              color = :blue, linewidth = 3,
              markersize = 12, marker = :circle)

scatterlines!(ax_cpu, nm_pure_cpu, t_pure_cpu,
              label = "FluxOptics (pure)",
              color = :orange, linewidth = 3,
              markersize = 12, marker = :utriangle)

scatterlines!(ax_cpu, nm_wave_cpu, t_wave_cpu,
              label = "WaveOpticsPropagation",
              color = :red, linewidth = 3,
              markersize = 12, marker = :rect)

scatterlines!(ax_cpu, nm_jax_cpu, t_jax_cpu,
              label = "JaxOptics",
              color = :green, linewidth = 3,
              markersize = 12, marker = :diamond)

scatterlines!(ax_cpu, nm_torch_cpu, t_torch_cpu,
              label = "TorchOptics",
              color = :purple, linewidth = 3,
              markersize = 12, marker = :star5)

axislegend(ax_cpu, position = :rb)

# Info box
info_text = """
Hardware: $(data_fo["hardware"]["gpu"]) | FFTW threads: $(data_fo["hardware"]["fftw_threads"]) | Resolution: 512×512
Max modes (GPU): Custom = $(maximum(nm_fo)), Pure = $(maximum(nm_pure)), WaveOpticsProp = $(maximum(nm_wave)), JaxOptics = $(maximum(nm_jax)), TorchOptics = $(maximum(nm_torch))
"""

Label(fig1[2, 1:2], info_text, tellwidth = false,
      fontsize = 14, halign = :left, valign = :top)

save(joinpath(results_dir, "comparison_performance.png"), fig1, px_per_unit = 2)
println("✅ Saved to results/comparison_performance.png")

# ============================================================================
# FIGURE 2: Speedup Analysis (vs FluxOptics custom)
# ============================================================================

fig2 = Figure(size = (1600, 800), fontsize = 18)

# GPU speedup - now with log scale to see TorchOptics values
ax_speedup_gpu = Axis(fig2[1, 1],
                      xlabel = "Number of modes",
                      ylabel = "Speedup vs FluxOptics (custom)",
                      title = "GPU Performance Comparison",
                      xscale = log10,
                      yscale = log10)

scatterlines!(ax_speedup_gpu, nm_speedup_pure, speedup_pure,
              label = "FluxOptics (pure)",
              color = :orange, linewidth = 3,
              markersize = 12, marker = :utriangle)

scatterlines!(ax_speedup_gpu, nm_speedup_wave, speedup_wave,
              label = "WaveOpticsPropagation",
              color = :red, linewidth = 3,
              markersize = 12, marker = :rect)

scatterlines!(ax_speedup_gpu, nm_speedup_jax, speedup_jax,
              label = "JaxOptics",
              color = :green, linewidth = 3,
              markersize = 12, marker = :diamond)

scatterlines!(ax_speedup_gpu, nm_speedup_torch, speedup_torch,
              label = "TorchOptics",
              color = :purple, linewidth = 3,
              markersize = 12, marker = :star5)

hlines!(ax_speedup_gpu, [1.0], color = :blue, linestyle = :dash, linewidth = 2,
        label = "FluxOptics (custom) baseline")

axislegend(ax_speedup_gpu, position = :rb)

ax_speedup_cpu = Axis(fig2[1, 2],
                      xlabel = "Number of modes",
                      ylabel = "Speedup vs FluxOptics (custom)",
                      title = "CPU Performance Comparison",
                      xscale = log10,
                      yscale = log10)

scatterlines!(ax_speedup_cpu, nm_speedup_pure_cpu, speedup_pure_cpu,
              label = "FluxOptics (pure)",
              color = :orange, linewidth = 3,
              markersize = 12, marker = :utriangle)

scatterlines!(ax_speedup_cpu, nm_speedup_wave_cpu, speedup_wave_cpu,
              label = "WaveOpticsPropagation",
              color = :red, linewidth = 3,
              markersize = 12, marker = :rect)

scatterlines!(ax_speedup_cpu, nm_speedup_jax_cpu, speedup_jax_cpu,
              label = "JaxOptics",
              color = :green, linewidth = 3,
              markersize = 12, marker = :diamond)

scatterlines!(ax_speedup_cpu, nm_speedup_torch_cpu, speedup_torch_cpu,
              label = "TorchOptics",
              color = :purple, linewidth = 3,
              markersize = 12, marker = :star5)

hlines!(ax_speedup_cpu, [1.0], color = :blue, linestyle = :dash, linewidth = 2,
        label = "FluxOptics (custom) baseline")

axislegend(ax_speedup_cpu, position = :rb)

# Info box
info_text2 = """
GPU - Pure: $(round(mean(speedup_pure), digits=2))×, WaveOpticsProp: $(round(mean(speedup_wave), digits=2))×, JaxOptics: $(round(mean(speedup_jax), digits=2))×, TorchOptics: $(round(mean(speedup_torch), digits=2))× ($(length(speedup_torch)) points)
CPU - Pure: $(round(mean(speedup_pure_cpu), digits=2))×, WaveOpticsProp: $(round(mean(speedup_wave_cpu), digits=2))×, JaxOptics: $(round(mean(speedup_jax_cpu), digits=2))×, TorchOptics: $(round(mean(speedup_torch_cpu), digits=2))× ($(length(speedup_torch_cpu)) points)
Note: Log scale on Y-axis to visualize TorchOptics slowdown (values < 1.0)
"""

Label(fig2[2, 1:2], info_text2, tellwidth = false,
      fontsize = 14, halign = :left, valign = :top)

save(joinpath(results_dir, "comparison_speedup.png"), fig2, px_per_unit = 2)
println("✅ Saved to results/comparison_speedup.png")
