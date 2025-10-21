# # Fox-Li Cavity Simulation: Quasi-Ince-Gaussian Mode Formation
#
# This tutorial demonstrates the Fox-Li iterative method for finding cavity eigenmodes
# in a semi-degenerate laser resonator configuration. Under specific conditions, the
# cavity supports **quasi-Ince-Gaussian modes** (quasi-IGᵉ₉₃ in this case) - modes 
# that are nearly invariant under propagation despite not being true IG eigenmodes.
#
# !!! note "Quasi-IG vs True IG Modes"
#     True Ince-Gaussian modes with closed nodal lines cannot be excited with single-lobe
#     pumping away from degeneracy: there will always exist a mode without closed nodal 
#     lines that has better overlap with the pump (gain-matching criterion). The quasi-IG 
#     patterns observed here only emerge near cavity degeneracies.
#     
#     For details, see: N. Barré, M. Romanelli, and M. Brunel, 
#     ["Role of cavity degeneracy for high-order mode excitation in end-pumped 
#     solid-state lasers"](https://doi.org/10.1364/OL.39.001022), Opt. Lett. **39**, 1022-1025 (2014).
#
# ## Physical Setup
#
# We simulate a laser cavity with:
# - A flat mirror with gain medium (near field)
# - A curved output coupler with aperture (far field)  
# - Cavity length close to the stability boundary (semi-degenerate)
#
# The Fox-Li method iteratively propagates the field through the cavity until
# convergence to the eigenmode.

using FluxOptics, CairoMakie
using CUDA  #src
CUDA.allowscalar(false)  #src
#nb using CUDA  # Comment if you don't have CUDA
using Random
Random.seed!(15);  # Determinist example
prefix = "01_FoxLi"  #src

# ## Initial Field: Speckle Pattern
#
# We start with a random speckle pattern as initial condition. This generic field
# contains contributions from all cavity modes - the iterative process will 
# select the dominant eigenmode through preferential amplification.

ns = 512, 512
ds = 4.0, 4.0

λ = 1.064
NA = 0.01

w0 = 500.0

speckle_dist = generate_speckle(ns, ds, λ, NA; envelope = Gaussian(w0))
u0 = ScalarField(speckle_dist, ds, λ)
u0 = cu(u0)  #src
#nb u0 = cu(u0)  # Comment if you don't have CUDA
normalize_power!(u0, 1e-1)

fig_speckle = visualize(u0, (intensity, complex);  #src
                        colormap = (:inferno, :dark), height = 120)  #src
save("docs/src/assets/$(prefix)_speckle.png", fig_speckle)  #src
#nb visualize(u0, (intensity, complex); colormap=(:inferno, :dark), height=200)
#md visualize(u0, (intensity, complex); colormap=(:inferno, :dark), height=120)
#md # ![Initial speckle field with Gaussian envelope](../assets/$(prefix)_speckle.png)

# ## Gain Medium Configuration
#
# The gain sheet provides selective amplification with saturable gain dynamics.
# The off-center Gaussian pump profile (xc = 140 µm) breaks the cylindrical symmetry.

dz = 1000.0
Isat = 1.0
g0m = 2e-3
wg = 30.0
xc = 140.0
gain = GainSheet(u0, dz, Isat, (x, y) -> g0m * exp(-((x-xc)^2+y^2)/wg^2))

fig_gain = visualize(gain, identity;  #src
                     colormap = :inferno, show_colorbars = true, height = 120)  #src
save("docs/src/assets/$(prefix)_gain.png", fig_gain)  #src
#nb visualize(gain, identity; colormap=:inferno, show_colorbars=true, height=200)
#md visualize(gain, identity; colormap=:inferno, show_colorbars=true, height=120)
#md # ![Gain profile - off-center Gaussian](../assets/$(prefix)_gain.png)

# ## Cavity Geometry: Semi-Degenerate Configuration
#
# The cavity length ℓ ≈ 101 mm is close to the half-degenerate condition (ℓ = Rc/2),
# where the stability parameter approaches unity. This near-instability regime supports
# complex transverse mode structures.
#
# We use a magnification factor of 2 between the two mirrors to properly sample
# the field at both ends of the cavity.

ℓ = 101000.0  # Close to half-degenerate cavity

s = 2
ds′ = s .* ds

p12 = ParaxialProp(u0, ds′, ℓ; use_cache = true)  # Magnification ×2
p21 = ParaxialProp(u0, ds′, ds, ℓ; use_cache = true)

half_cavity = gain |> p12

uf = half_cavity(u0).out

fig_speckle_mirror = visualize(uf, (intensity, complex);  #src
                               colormap = (:inferno, :dark), height = 120)  #src
save("docs/src/assets/$(prefix)_speckle_mirror.png", fig_speckle_mirror)  #src
#nb visualize(uf, (intensity, complex); colormap=(:inferno, :dark), height=200)
#md visualize(uf, (intensity, complex); colormap=(:inferno, :dark), height=120)
#md # ![Initial field at output mirror plane (magnified)](../assets/$(prefix)_speckle_mirror.png)

# ## Output Coupler: Spherical Mirror with Aperture
#
# The output mirror has 98% reflectivity and 200 mm radius of curvature.
# The aperture prevents the field from overflowing the computational grid,
# which would lead to numerical instabilities during iteration.

R = 0.98
Rc = 200000.0
aperture_radius = 1800.0
mirror = TeaReflector(uf, (x, y) -> -(x^2+y^2)/(2*Rc); r = -sqrt(R))
mirror_phase = phase(cis.(2π/λ .* get_data(mirror)))
aperture = Mask(uf, (x, y) -> x^2 + y^2 < aperture_radius^2 ? 1.0 : 0.0)

cavity = half_cavity |> mirror |> aperture |> p21 |> (; inplace = true)

fig_spherical_mirror = visualize(((mirror_phase, aperture),), real;  #src
                                 show_colorbars = true, height = 120)  #src
save("docs/src/assets/$(prefix)_spherical_mirror.png", fig_spherical_mirror)  #src
#nb visualize(((mirror_phase, aperture),), real; show_colorbars=true, height=200)
#md visualize(((mirror_phase, aperture),), real; show_colorbars=true, height=120)
#md # ![Spherical mirror phase profile (left) and circular aperture (right)](../assets/$(prefix)_spherical_mirror.png)

# ## Fox-Li Iteration: Eigenmode Convergence
#
# We iteratively propagate the field through the complete cavity round-trip.
# Each iteration consists of: gain → propagation → mirror+aperture → back-propagation.
# After ~5000 iterations, the field converges to the cavity eigenmode.

u0c = copy(u0)
utmp = u0c
#nb cavity(utmp);  # Warm-up for accurate timing

#nb @time for i in 1:5000
#md for i in 1:5000
for i in 1:5000  #src
    global utmp = cavity(utmp).out  #src
end  #src
#nb     utmp = cavity(utmp).out
#nb end
#md     utmp = cavity(utmp).out
#md end

# ## Converged Eigenmode: Quasi-Ince-Gaussian IGᵉ₉₃
#
# The converged mode exhibits an elliptical intensity pattern reminiscent of 
# Ince-Gaussian modes. This quasi-IG mode only exists near the cavity degeneracy point
# and maintains its structure approximately through propagation.

u_near = cavity(utmp).out
u_mirror = half_cavity(u_near).out

fig_eigenmode = visualize((u_near, u_mirror), (intensity, phase);  #src
                          colormap = (:inferno, :twilight), show_colorbars = true,  #src
                          height = 150)  #src
save("docs/src/assets/$(prefix)_eigenmode.png", fig_eigenmode)  #src
#nb visualize((u_near, u_mirror), (intensity, phase); colormap=(:inferno, :twilight), show_colorbars=true, height=300)
#md visualize((u_near, u_mirror), (intensity, phase);
#md            colormap=(:inferno, :twilight), show_colorbars=true, height=150)
#md # ![Converged eigenmode at near field (left) and far field mirror (right)](../assets/$(prefix)_eigenmode.png)

# ## Far-Field Characterization
#
# Propagating the eigenmode to twice the cavity length reveals the far-field intensity
# distribution and phase structure.

p_far = ParaxialProp(u_near, 2 .* ds, 2*ℓ; use_cache = true)
u_far = propagate(u_near, p_far, Forward)

fig_far = visualize(u_far, (intensity, phase);  #src
                    colormap = (:inferno, :twilight), show_colorbars = true, height = 150)  #src
save("docs/src/assets/$(prefix)_far.png", fig_far)  #src
#nb visualize(u_far, (intensity, phase); colormap=(:inferno, :twilight), show_colorbars=true, height=300)
#md visualize(u_far, (intensity, phase);
#md            colormap=(:inferno, :twilight), show_colorbars=true, height=150)
#md # ![Far-field intensity and phase at 2ℓ distance](../assets/$(prefix)_far.png)
