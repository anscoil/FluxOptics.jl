using FluxOptics

Prop = ASPropZ
include("fluxoptics_common.jl")

println("✅ FluxOptics pure AS correct!")

n_bench = 20
nmodes_list = Int.(round.(range(1, n_modes, length = n_bench)))

bench_cpu_results = []
bench_gpu_results = []

println("Benchmarking $(length(nmodes_list)) configurations...")
println("="^60)

for (i, n_modes) in enumerate(nmodes_list)
    print("[$i/$(length(nmodes_list))] $n_modes modes: ")
    GC.gc()
    CUDA.reclaim()

    # CPU
    print("CPU...")
    local bench_cpu = bench_propagate_cpu(input_modes, n_modes, dx, dy, λ, z)
    b = @benchmark $bench_cpu() seconds=10
    cpu_time = median(b.times) / 1e6
    cpu_deviation_time = std(b.times) / 1e6
    @assert !isnan(cpu_time) && !isnan(cpu_deviation_time)
    push!(bench_cpu_results,
          Dict("n_modes" => n_modes,
               "time_median_ms" => cpu_time,
               "time_std_ms" => cpu_deviation_time))

    # GPU
    print(" GPU...")
    try
        local bench_gpu = bench_propagate_gpu(input_modes, n_modes, dx, dy, λ, z)
        b = @benchmark CUDA.@sync $bench_gpu()
        bench_gpu = nothing
        gpu_time = median(b.times) / 1e6
        gpu_deviation_time = std(b.times) / 1e6
        @assert !isnan(gpu_time) && !isnan(gpu_deviation_time)
        push!(bench_gpu_results,
              Dict("n_modes" => n_modes,
                   "time_median_ms" => gpu_time,
                   "time_std_ms" => gpu_deviation_time))
        @printf " ✓ (CPU: %.2f ms, GPU: %.2f ms, speedup: %.1f×)\n" cpu_time gpu_time cpu_time/gpu_time
    catch e
        println(" ✗ GPU Out Of Memory")
    end
end

println("="^60)
println("✅ Done!")

# Combine results + metadata
output = Dict("cpu" => bench_cpu_results,
              "gpu" => bench_gpu_results,
              "config" => Dict("wavelength" => λ,
                               "z" => z,
                               "dx" => dx,
                               "dy" => dy,
                               "n_bench" => n_bench),
              "hardware" => Dict("gpu" => CUDA.name(CUDA.device()),
                                 "fftw_threads" => FFTW.get_num_threads()))

# Save
results_dir = joinpath(@__DIR__, "../results")
mkpath(results_dir)

open(joinpath(results_dir, "fluxoptics_pure_bench.json"), "w") do f
    JSON.print(f, output, 2)
end

println("✅ Saved to results/fluxoptics_pure_bench.json")
