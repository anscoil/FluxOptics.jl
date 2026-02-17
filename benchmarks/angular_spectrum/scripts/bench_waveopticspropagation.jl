using LinearAlgebra
using WaveOpticsPropagation
using BenchmarkTools
using NPZ
using JSON
using Printf
using CUDA  # Comment if you don't have CUDA
using FFTW
FFTW.set_num_threads(Sys.CPU_THREADS)

function Base.vec(u::AbstractArray, nd::Integer)
    @assert nd in (1, 2)
    @assert ndims(u) >= nd
    reshape(eachslice(u; dims = Tuple((nd + 1):ndims(u))), :)
end

path = joinpath(@__DIR__, "../data/test_cases.npz")

data = npzread(path)
input_modes = data["input_modes"]
output_modes = data["output_modes"]
n_modes_orig = size(input_modes, 3)

λ = data["wavelength"]
z = data["z"]
dx = data["dx"]
dy = data["dy"]

nx = size(input_modes, 1)

# Data augmentation to reach limit of memory usage
N_rep = 6
n_modes = n_modes_orig * N_rep
input_modes = repeat(input_modes, 1, 1, N_rep)
output_modes = repeat(output_modes, 1, 1, N_rep)

# Check correctness
function check_correctness(uin, uout, nx, dx, λ, z)
    p_fwd = AngularSpectrum(uin[1], z, λ, nx*dx; padding = false, bandlimit = false)
    p_bwd = AngularSpectrum(uin[1], -z, λ, nx*dx; padding = false, bandlimit = false)

    function correct()
        uf = [copy(p_fwd(x; crop = false)) for x in uin]
        ub = [copy(p_bwd(x; crop = false)) for x in uf]
        (all(isapprox.(abs2.(dot.(uf, uout)), 1; atol = 1e-4)),
         all(isapprox.(abs2.(dot.(ub, uin)), 1; atol = 1e-4)))
    end
    correct
end

uin = vec(input_modes, 2)
uout = vec(output_modes, 2)

fwd_correct, bwd_correct = check_correctness(uin, uout, nx, dx, λ, z)()
@assert fwd_correct
@assert bwd_correct

println("✅ WaveOpticsPropagation AS correct!")

function bench_propagate!(uin, nx, dx, λ, z)
    p_fwd = AngularSpectrum(uin[1], z, λ, nx*dx; padding = false, bandlimit = false)
    p_bwd = AngularSpectrum(uin[1], -z, λ, nx*dx; padding = false, bandlimit = false)

    function prop()
        for x in uin
            uf = p_fwd(x; crop = false)
            p_bwd(uf; crop = false)
        end
    end
    prop
end

function bench_propagate(uin, nx, dx, λ, z)
    bench_propagate!(uin, nx, dx, λ, z)
end

n_bench = 20
# nmodes_list = Int.(round.(range(1, n_modes, length = n_bench)))
nmodes_list = unique([1;
                      Int.(round.(exp.(range(log(10), log(n_modes), length = n_bench))))])

bench_cpu_results = []
bench_gpu_results = []

println("Benchmarking $(length(nmodes_list)) configurations...")
println("="^60)

for (i, n_modes) in enumerate(nmodes_list)
    print("[$i/$(length(nmodes_list))] $n_modes modes: ")

    # CPU
    print("CPU...")
    local bench_cpu = bench_propagate(@view(uin[1:n_modes]), nx, dx, λ, z)
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
        ui = vec(cu(@view(input_modes[:, :, 1:n_modes])), 2)
        local bench_gpu = bench_propagate(ui, nx, dx, λ, z)
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

open(joinpath(results_dir, "waveopticspropagation_bench.json"), "w") do f
    JSON.print(f, output, 2)
end

println("✅ Saved to results/waveopticspropagation_bench.json")
