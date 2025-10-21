# # Field Retrieval: Reconstructing Complex Fields from Intensity Measurements
#
# This tutorial demonstrates phase retrieval using gradient-based optimization.
# From intensity-only measurements in two planes, we reconstruct the full complex
# field (amplitude and phase) using automatic differentiation.
#
# ## The Phase Problem
#
# Standard detectors measure only intensity |u|², losing phase information. However,
# with measurements at multiple propagation distances, the complex field can be
# recovered through optimization - a generalization of the Gerchberg-Saxton algorithm
# using gradients rather than alternating projections.

using FluxOptics, Zygote, CairoMakie
using CUDA  #src
CUDA.allowscalar(false)  #src
#nb using CUDA  # Comment if you don't have CUDA
using Random
Random.seed!(15);  # Determinist example
prefix = "02_field_retrieval"  #src

# ## Ground Truth: Laguerre-Gaussian Mode
#
# We create a high-order LG mode as our target field. This will serve as the
# ground truth - we'll measure its intensity at two planes, then attempt to
# reconstruct the complex field from these measurements alone.

ns = 512, 512
ds = 2.0, 2.0
x_vec, y_vec = spatial_vectors(ns, ds)

λ = 1.064
z1 = 5000.0
z2 = 15000.0

w0 = 50.0
m, n = 2, 3

u0_th = LaguerreGaussian(w0, m, n; kind = :odd)(x_vec, y_vec)
u1_th = LaguerreGaussian(w0, m, n, λ, z1; kind = :odd)(x_vec, y_vec)
u2_th = LaguerreGaussian(w0, m, n, λ, z2; kind = :odd)(x_vec, y_vec)

fig_ground_truth = visualize(((u0_th, u1_th, u2_th),), intensity;  #src
                             colormap = :inferno, height = 120)  #src
save("docs/src/assets/$(prefix)_ground_truth.png", fig_ground_truth)  #src
#nb visualize(((u0_th, u1_th, u2_th),), intensity; colormap=:inferno, height=200)
#md visualize(((u0_th, u1_th, u2_th),), intensity; colormap=:inferno, height=120)
#md # ![Ground truth LG mode at z=0, z₁, and z₂](../assets/02_field_retrieval_ground_truth.png)

# ## Intensity Measurements
#
# In a real experiment, we would measure only the intensity at two planes.
# Here we simulate these measurements from our known ground truth.

I1 = abs.(u1_th) .^ 2
I2 = abs.(u2_th) .^ 2

@assert isapprox(sum(I1)*prod(ds), 1; atol = 1e-7)  # Power normalized to unity
@assert isapprox(sum(I2)*prod(ds), 1; atol = 1e-7)  # Power normalized to unity

I1 = cu(I1)  #src
I2 = cu(I2)  #src
#nb I1 = cu(I1)  # Comment if you don't have CUDA
#nb I2 = cu(I2)  # Comment if you don't have CUDA

fig_measurements = visualize(((I1, I2),), real;  #src
                             colormap = :inferno, height = 120)  #src
save("docs/src/assets/$(prefix)_measurements.png", fig_measurements)  #src
#nb visualize(((I1, I2),), real; colormap=:inferno, height=200)
#md visualize(((I1, I2),), real; colormap=:inferno, height=120)
#md # ![Intensity measurements at z₁=5mm and z₂=15mm](../assets/02_field_retrieval_measurements.png)

# ## Initial Guess: Random Speckle
#
# We start from a random speckle pattern - a generic field with no knowledge
# of the target mode. The optimization will transform this into the LG mode.

NA = 0.01
speckle_dist = generate_speckle(ns, ds, λ, NA; envelope = Gaussian(3*w0))
u0 = ScalarField(speckle_dist, ds, λ)
u0 = cu(u0)  #src
#nb u0 = cu(u0)  # Comment if you don't have CUDA

fig_init = visualize(u0, (intensity, complex);  #src
                     colormap = (:inferno, :dark), height = 120)  #src
save("docs/src/assets/$(prefix)_initial.png", fig_init)  #src
#nb visualize(u0, (intensity, complex); colormap=(:inferno, :dark), height=200)
#md visualize(u0, (intensity, complex); colormap=(:inferno, :dark), height=120)
#md # ![Initial random speckle field](../assets/02_field_retrieval_initial.png)

# ## Optical System Setup
#
# We construct the propagation system with probes to capture fields at measurement
# planes. The source field is trainable - its complex values will be optimized.

p1 = RSProp(u0, z1; use_cache = true)
p2 = RSProp(u0, z2-z1; use_cache = true)

s = ScalarSource(u0; trainable = true, buffered = true)
fp1 = FieldProbe()
fp2 = FieldProbe()

model = s |> p1 |> fp1 |> p2 |> fp2 |> (; inplace = true);

# ## Loss Function
#
# We minimize the difference between predicted and measured intensities using
# a correlation-based metric. The loss is defined as 2 - (correlation₁ + correlation₂),
# where perfect reconstruction gives loss = 0.

sI1 = sqrt.(I1)
sI2 = sqrt.(I2)
eps = 1e-15 * maximum(I1)

function f_opt(m)
    _, probes = m()
    u1 = probes[fp1]
    u2 = probes[fp2]
    (2 -
     (dot(sqrt.(abs2.(u1.electric) .+ eps), sI1) +
      dot(sqrt.(abs2.(u2.electric) .+ eps), sI2)) * prod(ds))
end

# ## Optimization
#
# We use FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) with gradient-based
# optimization. The non-convex nature of phase retrieval means convergence can take
# many iterations, with the optimizer navigating through local minima. Power
# normalization at each step prevents divergence.

opt = setup(Fista(1), model)
_, g = Zygote.withgradient(f_opt, model)  # Warm-up  #src
FluxOptics.update!(opt, model, g[1])  # Warm-up  #src
#nb _, g = Zygote.withgradient(f_opt, model);  # Warm-up
#nb FluxOptics.update!(opt, model, g[1]);  # Warm-up

fill!(s, u0)
losses = []

#nb @time for i in 1:1000
#md for i in 1:1000
for i in 1:1000  #src
    val, grads = Zygote.withgradient(f_opt, model)  #src
    FluxOptics.update!(opt, model, grads[1])  #src
    normalize_power!(get_source(s), 1)  # Power normalization to avoid divergence  #src
    push!(losses, val)  #src
end  #src
#nb     val, grads = Zygote.withgradient(f_opt, model)
#nb     FluxOptics.update!(opt, model, grads[1])
#nb     normalize_power!(get_source(s), 1)  # Power normalization to avoid divergence
#nb     push!(losses, val)
#nb end
#md     val, grads = Zygote.withgradient(f_opt, model)
#md     FluxOptics.update!(opt, model, grads[1])
#md     normalize_power!(get_source(s), 1)  # Power normalization to avoid divergence
#md     push!(losses, val)
#md end

# The convergence curve reveals the non-convex nature of the phase retrieval problem,
# with several plateaus before reaching the global minimum.

fig_loss = Figure(size = (350, 280))  #src
ax = Makie.Axis(fig_loss[1, 1], yscale = log10, xlabel = "Iteration", ylabel = "Loss")  #src
lines!(ax, losses; linewidth = 2)  #src
save("docs/src/assets/$(prefix)_convergence.png", fig_loss)  #src
#nb fig_loss = Figure()
#nb ax = Makie.Axis(fig_loss[1, 1], yscale = log10, xlabel = "Iteration", ylabel = "Loss")
#nb lines!(ax, losses; linewidth = 4)
#nb fig_loss
#md # ![Convergence showing non-convex optimization landscape with multiple plateaus](../assets/02_field_retrieval_convergence.png)

# ## Reconstructed Field
#
# After optimization, we extract the reconstructed field and propagated versions
# at the measurement planes. The complex field structure (intensity and phase)
# is successfully recovered from intensity-only measurements.

_, probes = model()
u1 = probes[fp1]
u2 = probes[fp2]

fig_result = visualize((s, u1, u2), (intensity, phase);  #src
                       colormap = (:inferno, :twilight), show_colorbars = true,  #src
                       height = 150)  #src
save("docs/src/assets/$(prefix)_result.png", fig_result)  #src
#nb visualize((s, u1, u2), (intensity, phase); colormap=(:inferno, :twilight), show_colorbars=true, height=300)
#md visualize((s, u1, u2), (intensity, phase);
#md            colormap=(:inferno, :twilight), show_colorbars=true, height=150)
#md # ![Reconstructed field at z=0, z₁, and z₂ - intensity (top) and phase (bottom)](../assets/02_field_retrieval_result.png)

# ## Reconstruction Quality
#
# We quantify reconstruction quality using coupling efficiency - the overlap
# integral between reconstructed and true fields. Values above 99% indicate
# excellent reconstruction.

coupling_effs = [coupling_efficiency(u, v)  #src
                 for (u, v) in  #src
                     zip(collect.((s, u1, u2)), (u0_th, u1_th, u2_th))]  #src
println("Coupling efficiencies: ", coupling_effs)  #src
#nb coupling_effs = [coupling_efficiency(u, v) for (u, v) in 
#nb                  zip(collect.((s, u1, u2)), (u0_th, u1_th, u2_th))]
#md # | Plane | Coupling Efficiency |
#md # |-------|---------------------|
#md # | z = 0 | 99.995% |
#md # | z₁ = 5 mm | 99.995% |
#md # | z₂ = 15 mm | 99.995% |
#md # 
#md # The reconstruction achieves >99.99% coupling efficiency, demonstrating accurate
#md # recovery of both amplitude and phase from intensity measurements alone.
#md #
#md # !!! note "Extension to Noisy Measurements"
#md #     With realistic measurement noise, regularization techniques such as Total
#md #     Variation (TV) or support constraints become necessary. The core optimization
#md #     framework remains the same, with proximal operators added to the optimization rule.
