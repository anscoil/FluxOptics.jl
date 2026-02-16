import torch
import torchoptics
from torchoptics import Field
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import time
import json
from statistics import median, stdev

torch.set_default_dtype(torch.float32)

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
N_rep = 1
n_modes = n_modes_orig * N_rep

input_modes = np.tile(input_modes, (N_rep, 1, 1))
output_modes = np.tile(output_modes, (N_rep, 1, 1))

def dtype(device):
    return torch.complex128 if device == "cpu" else torch.complex64

def field(data, i, device="cpu"):
    field = torch.tensor(data[i], device=device, dtype=dtype(device))
    return Field(field, wavelength, spacing=(dx, dy)).to(device)

def propagate(u, z):
    return u.propagate_to_z(z, propagation_method="ASM")

# Check correctness
def check_overlap(fields1, fields2):
    """fields1 and fields2 are lists of Field objects"""
    overlaps_np = [(torch.abs(torch.sum(torch.conj(f1.data) * f2.data))**2).item() 
                   for f1, f2 in zip(fields1, fields2)]
    return all(abs(x - 1.0) < 1e-4 for x in overlaps_np)

def check_correctness(uin, uout, z):
    def correct():
        uf = []
        ub = []
        for u0 in uin:
            ufi = propagate(u0, z)
            ubi = propagate(ufi, 0)
            uf.append(ufi)
            ub.append(ubi)
        return (check_overlap(uf, uout), check_overlap(ub, uin))
    return correct

# device = "cpu"

# # Create Fields for each mode
# uin = [field(input_modes, i, device) for i in range(n_modes_orig)]
# uout = [field(output_modes, i, device) for i in range(n_modes_orig)]

# # Check correctness
# fwd_correct, bwd_correct = check_correctness(uin, uout, z)()
# assert fwd_correct, "Forward propagation correctness check failed"
# assert bwd_correct, "Backward propagation correctness check failed"
# print("✅ TorchOptics AS correct!")


def benchmark_torch(input_modes, n_modes, dx, dy, wavelength, z, device='cuda'):
    # Setup - create list of Fields
    u_list = [field(input_modes, i, device) for i in range(n_modes)]
    
    def propagate_roundtrip(fields):
        uf = [propagate(u, z) for u in fields]
        ub = [propagate(u, 0) for u in uf]
        return ub
    
    # Warmup
    result = propagate_roundtrip(u_list)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(50):
        start = time.perf_counter()
        result = propagate_roundtrip(u_list)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed_time = (time.perf_counter() - start) * 1000
        times.append(elapsed_time)
        if elapsed_time > 15000:  # 15 seconds
            break
    
    return {
        "n_modes": int(n_modes),
        "time_median_ms": float(median(times)),
        "time_std_ms": float(stdev(times)),
    }

# Generate log-spaced nmodes list
n_bench = 5
n_modes = 30
nmodes_list = np.round(np.linspace(1, n_modes, n_bench)).astype(int)

bench_cpu_results = []
bench_gpu_results = []

print(f"Benchmarking {len(nmodes_list)} configurations...")
print("=" * 60)

for i, nm in enumerate(nmodes_list):
    print(f"[{i+1}/{len(nmodes_list)}] {nm} modes: ", end="")
    
    # CPU with try-catch
    print("CPU...", end="", flush=True)
    cpu_success = False
    try:
        cpu_result = benchmark_torch(input_modes, nm, dx, dy, wavelength, z, device='cpu')
        cpu_time = cpu_result["time_median_ms"]
        assert not np.isnan(cpu_time)
        bench_cpu_results.append(cpu_result)
        cpu_success = True
    except Exception as e:
        print(f" ✗ CPU Error", end="")
        cpu_time = None
    
    # GPU with try-catch
    print(" GPU...", end="", flush=True)
    try:
        gpu_result = benchmark_torch(input_modes, nm, dx, dy, wavelength, z, device='cuda')
        gpu_time = gpu_result["time_median_ms"]
        assert not np.isnan(gpu_time)
        bench_gpu_results.append(gpu_result)
        
        if cpu_success:
            speedup = cpu_time / gpu_time
            print(f" ✓ (CPU: {cpu_time:.2f} ms, GPU: {gpu_time:.2f} ms, speedup: {speedup:.1f}×)")
        else:
            print(f" ✓ (GPU: {gpu_time:.2f} ms)")
    except Exception as e:
        print(f" ✗ GPU Error")

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
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "cpu_count": 1,
    },
}

# Save
results_dir = Path(__file__).parent / "../results"
results_dir.mkdir(parents=True, exist_ok=True)

output_path = results_dir / "torchoptics_bench.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"✅ Saved to {output_path}")
