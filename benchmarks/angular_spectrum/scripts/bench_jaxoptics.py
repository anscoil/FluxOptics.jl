import numpy as np
import jax
import jax.numpy as jnp
from jaxoptics import *
from pathlib import Path
import time
import json
from statistics import median, stdev

script_dir = Path(__file__).parent
path = script_dir / "../data/test_cases.npz"

data = np.load(path)
input_modes = data["input_modes"].T
output_modes = data["output_modes"].T
n_modes_orig = input_modes.shape[0]

wavelength = float(data["wavelength"])
z = float(data["z"])
dx = float(data["dx"])
dy = float(data["dy"])

# Data augmentation
N_rep = 6
n_modes = n_modes_orig * N_rep

input_modes = np.tile(input_modes, (N_rep, 1, 1))
output_modes = np.tile(output_modes, (N_rep, 1, 1))

# Check correctness
def check_overlap(field1, field2):
    e1 = field1.electric
    e2 = field2.electric
    overlap = jnp.sum(jnp.conj(e1) * e2, axis=(1, 2))
    overlap_squared = jnp.abs(overlap)**2
    
    return jnp.all(jnp.isclose(overlap_squared, 1.0, atol=1e-4))

def check_correctness(uin, uout, z):
    p_fwd = ASProp(uin, z, trainable=False)
    p_bwd = ASProp(uin, -z, trainable=False)
    def correct():
        uf = p_fwd(uin)
        ub = p_bwd(uf)
        return (check_overlap(uf, uout), check_overlap(ub, uin))
    return correct

with jax.default_device(jax.devices('cpu')[0]):
    uin = jnp.array(input_modes[1:n_modes_orig])
    uout = jnp.array(output_modes[1:n_modes_orig])
    u0 = ScalarField(uin, (dx, dy), wavelength)
    vf = ScalarField(uout, (dx, dy), wavelength)
    fwd_correct, bwd_correct = check_correctness(u0, vf, z)()
    assert fwd_correct, "Forward propagation correctness check failed"
    assert bwd_correct, "Backward propagation correctness check failed"
    print("✅ JaxOptics AS correct!")

def benchmark_jax(input_modes, n_modes, dx, dy, wavelength, z, device='gpu'):
    # Setup
    with jax.default_device(jax.devices(device)[0]):
        uin = jnp.array(input_modes[:n_modes])
        u = ScalarField(uin, (dx, dy), wavelength)
        p_fwd = ASProp(u, z, trainable=False)
        p_bwd = ASProp(u, -z, trainable=False)
        
        # JIT compiled function
        @jax.jit
        def propagate_roundtrip(field):
            uf = p_fwd(field)
            ub = p_bwd(uf)
            return ub
        
        # Warmup (compile + cache)
        jax.block_until_ready(propagate_roundtrip(u))
        
        # Benchmark
        times = []
        for _ in range(50):
            start = time.perf_counter()
            result = propagate_roundtrip(u)
            result.electric.block_until_ready()
            elapsed_time = (time.perf_counter() - start) * 1000
            times.append(elapsed_time)
            if elapsed_time > 10000:  # 10 seconds = 10000 ms
                break

        return {
            "n_modes": int(n_modes),
            "time_median_ms": float(median(times)),
            "time_std_ms": float(stdev(times)),
        }

# Generate log-spaced nmodes list
n_bench = 20
nmodes_list = np.unique(np.concatenate([
    [1],
    np.round(np.exp(np.linspace(np.log(10), np.log(n_modes), n_bench))).astype(int)
]))

bench_cpu_results = []
bench_gpu_results = []

print(f"Benchmarking {len(nmodes_list)} configurations...")
print("=" * 60)

for i, nm in enumerate(nmodes_list):
    print(f"[{i+1}/{len(nmodes_list)}] {nm} modes: ", end="")
    
    # CPU
    print("CPU...", end="", flush=True)
    cpu_result = benchmark_jax(input_modes, nm, dx, dy, wavelength, z, device='cpu')
    cpu_time = cpu_result["time_median_ms"]
    assert not np.isnan(cpu_time)
    bench_cpu_results.append(cpu_result)
    
    # GPU
    print(" GPU...", end="", flush=True)
    try:
        gpu_result = benchmark_jax(input_modes, nm, dx, dy, wavelength, z, device='gpu')
        gpu_time = gpu_result["time_median_ms"]
        assert not np.isnan(gpu_time)
        bench_gpu_results.append(gpu_result)
        
        speedup = cpu_time / gpu_time
        print(f" ✓ (CPU: {cpu_time:.2f} ms, GPU: {gpu_time:.2f} ms, speedup: {speedup:.1f}×)")
    except Exception as e:
        print(f" ✗ GPU Out Of Memory")

print("=" * 60)
print("✅ Done!")

# Combine results + metadata
output = {
    "cpu": bench_cpu_results,
    "gpu": bench_gpu_results,
    "config": {
        "wavelength": float(wavelength),
        "z": float(z),
        "dx": float(dx),
        "dy": float(dy),
        "n_bench": n_bench,
    },
    "hardware": {
        "gpu": jax.devices('gpu')[0].device_kind if jax.devices('gpu') else "N/A",
        "cpu_count": len(jax.devices('cpu')),
    },
}

# Save
results_dir = Path(__file__).parent / "../results"
results_dir.mkdir(parents=True, exist_ok=True)

output_path = results_dir / "jaxoptics_bench.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"✅ Saved to {output_path}")
