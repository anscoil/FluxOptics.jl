using BenchmarkTools
using NPZ
using JSON
using Printf
using CUDA  # Comment if you don't have CUDA
using FFTW
FFTW.set_num_threads(Sys.CPU_THREADS)

path = joinpath(@__DIR__, "../data/test_cases.npz")

data = npzread(path)
input_modes = data["input_modes"]
output_modes = data["output_modes"]
n_modes_orig = size(input_modes, 3)

λ = data["wavelength"]
z = data["z"]
dx = data["dx"]
dy = data["dy"]

# Data augmentation to reach limit of memory usage
N_rep = 6
n_modes = n_modes_orig * N_rep
input_modes = repeat(input_modes, 1, 1, N_rep)
output_modes = repeat(output_modes, 1, 1, N_rep)

u0 = ScalarField(input_modes[:, :, 1:n_modes_orig], (dx, dy), λ)
vf = ScalarField(output_modes[:, :, 1:n_modes_orig], (dx, dy), λ)

u0_gpu = cu(u0)  # Comment if you don't have CUDA
vf_gpu = cu(vf)  # Comment if you don't have CUDA

# Check correctness
function check_correctness(uin, uout, z)
    p = Prop(uin, z)
    function correct()
        uf = propagate(uin, p, Forward)
        ub = propagate(uf, p, Backward)
        (all(isapprox.(abs2.(dot(uf, uout)), 1; atol = 1e-4)),
         all(isapprox.(abs2.(dot(ub, uin)), 1; atol = 1e-4)))
    end
    correct
end

fwd_correct, bwd_correct = check_correctness(u0, vf, z)()
@assert fwd_correct
@assert bwd_correct

fwd_correct, bwd_correct = check_correctness(u0_gpu, vf_gpu, z)()  # Comment if you don't have CUDA
@assert fwd_correct
@assert bwd_correct

function bench_propagate!(u, z)
    p = Prop(u, z)
    function prop()
        propagate!(u, p, Forward)
        propagate!(u, p, Backward)
    end
    prop
end

function bench_propagate_cpu(input_modes, n_modes, dx, dy, λ, z)
    u = ScalarField(input_modes[:, :, 1:n_modes], (dx, dy), λ)
    bench_propagate!(u, z)
end

function bench_propagate_gpu(input_modes, n_modes, dx, dy, λ, z)
    u = cu(ScalarField(input_modes[:, :, 1:n_modes], (dx, dy), λ))
    bench_propagate!(u, z)
end
