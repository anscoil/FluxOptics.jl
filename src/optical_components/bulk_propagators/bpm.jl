const BPMProp = Union{typeof(ASProp), typeof(ShiftProp)}

function compute_cos_correction(u::ScalarField)
    θs = get_tilts(u)
    reduce((.*), map(x -> cos.(x), θs))
end

struct BPM{M, A, U, D, P, K} <: AbstractCustomComponent{M}
    dn::A
    kdz::K
    aperture_mask::D
    p_bpm::P
    p_bpm_half::P
    ∂p::Union{Nothing, @NamedTuple{dn::A}}
    u::Union{Nothing, U}

    function BPM(dn::A,
                 kdz::K,
                 aperture_mask::D,
                 p_bpm::P,
                 p_bpm_half::P,
                 ∂p::Union{Nothing, @NamedTuple{dn::A}},
                 u::U) where {A, K, D, P, U}
        M = isnothing(u) ? Trainable{Unbuffered} : Trainable{Buffered}
        new{M, A, U, D, P, K}(dn, kdz, aperture_mask, p_bpm, p_bpm_half, ∂p, u)
    end

    function _init(u::U,
                   ds::NTuple{Nd, Real},
                   thickness::Real,
                   dn0::AbstractArray{<:Real, Nv},
                   trainable::Bool,
                   buffered::Bool,
                   aperture::Function) where {Nd, Nv, N, T,
                                              U <: AbstractArray{Complex{T}, N}}
        @assert Nd in (1, 2)
        @assert Nv == Nd + 1
        @assert N >= Nd
        n_slices = size(dn0, Nv)
        @assert n_slices >= 2
        dz = thickness / n_slices
        M = trainability(trainable, buffered)
        A = similar(U, real, Nv)
        D = similar(U, real, Nd)
        dn = A(dn0)
        xs = spatial_vectors(size(u)[1:Nd], ds)
        aperture_mask = Nd == 2 ? D(aperture.(xs[1], xs[2]')) : D(aperture.(xs[1]))
        ∂p = (trainable && buffered) ? (; dn = similar(dn)) : nothing
        u = (trainable && buffered) ? similar(u, (size(u)..., n_slices)) : nothing
        Us = similar(U, N+1)
        ((M, A, Us, D), (dn, dz, aperture_mask, ∂p, u))
    end

    function BPM(Prop::BPMProp,
                 use_cache::Bool,
                 u::ScalarField,
                 thickness::Real,
                 dn0::AbstractArray{<:Real};
                 trainable::Bool = false,
                 buffered::Bool = false,
                 aperture::Function = (_...) -> 1,
                 double_precision_kernel::Bool = use_cache,
                 kwargs = (;))
        ((M, A, U, D),
         (dn, dz, aperture_mask, ∂p, u_saved)) = _init(u.electric, Tuple(u.ds), thickness,
                                                       dn0, trainable, buffered, aperture)
        p_bpm = Prop(u, dz; use_cache, double_precision_kernel, kwargs...)
        p_bpm_half = Prop(u, dz/2; use_cache, double_precision_kernel, kwargs...)
        P = typeof(p_bpm)
        kdz = (2π*dz) ./ compute_cos_correction(u)
        K = typeof(kdz)
        new{M, A, U, D, P, K}(dn, kdz, aperture_mask, p_bpm, p_bpm_half, ∂p, u_saved)
    end
end

"""
    AS_BPM(u::ScalarField, thickness, n0, dn0; use_cache=true, paraxial=false, trainable=false, buffered=false, aperture=(x,y)->1, double_precision_kernel=use_cache)

Beam Propagation Method using Angular Spectrum for inhomogeneous media.

Propagates through a volume with spatially-varying refractive index using split-step
method with Angular Spectrum propagation. Handles both on-axis and tilted beams.

# Arguments
- `u::ScalarField`: Field template
- `thickness::Real`: Total propagation distance
- `n0::Real`: Background refractive index
- `dn0::AbstractArray{<:Real}`: Refractive index variation (3D: nx × ny × n_slices)
- `use_cache::Bool`: Cache propagation kernels (default: true)
- `paraxial::Bool`: Use paraxial approximation (default: false)
- `trainable::Bool`: Optimize refractive index profile (default: false)
- `buffered::Bool`: Pre-allocate gradient buffers (default: false)
- `aperture::Function`: Aperture function (x, y) -> transmission (default: unity)
- `double_precision_kernel::Bool`: Use Float64 for kernels (default: use_cache)

# Physics

Split-step method:
1. Half-step propagation in background (n₀)
2. Apply phase shift: exp(i k₀ Δn(x,y,z) Δz)
3. Full-step propagation in background
4. Repeat for all slices

Phase correction includes cosine factor for tilted beams to account for oblique
propagation geometry.

# Examples
```julia
u = ScalarField(ones(ComplexF64, 256, 256), (2.0, 2.0), 1.064)

# Uniform refractive index variation
thickness = 1000.0  # μm
n_slices = 100
dn = 0.01 * ones(256, 256, n_slices)  # Constant Δn
bpm = AS_BPM(u, thickness, 1.0, dn)

# Graded-index fiber
r = sqrt.(xv.^2 .+ yv'.^2)
dn_fiber = -0.01 * (r/50).^2  # Parabolic index
dn_3d = repeat(dn_fiber, 1, 1, n_slices)
bpm_fiber = AS_BPM(u, thickness, 1.5, dn_3d)

# Trainable refractive index (e.g., waveguide design)
dn_init = zeros(256, 256, n_slices)
bpm_opt = AS_BPM(u, thickness, 1.0, dn_init; trainable=true, buffered=true)
```

See also: [`Shift_BPM`](@ref), [`ASProp`](@ref)
"""
function AS_BPM(u::ScalarField,
                thickness::Real,
                n0::Real,
                dn0::AbstractArray{<:Real};
                use_cache::Bool = true,
                paraxial::Bool = false,
                trainable::Bool = false,
                buffered::Bool = false,
                aperture::Function = (_...) -> 1,
                double_precision_kernel::Bool = use_cache)
    BPM(ASProp, use_cache, u, thickness, dn0; trainable, buffered, aperture,
        double_precision_kernel, kwargs = (; n0, paraxial))
end

"""
    Shift_BPM(u::ScalarField, thickness, dn0; use_cache=true, trainable=false, buffered=false, aperture=(x,y)->1, double_precision_kernel=use_cache)

Beam Propagation Method using geometric shift (no diffraction).

Propagates through inhomogeneous media using pure geometric shifts based on field tilts.
Equivalent to backprojection in tomography. Useful for comparison with diffraction-based
methods or for high-NA/short-wavelength regimes where ray optics dominates.

# Arguments
- `u::ScalarField`: Field template (must have tilts defined)
- `thickness::Real`: Total propagation distance
- `dn0::AbstractArray{<:Real}`: Refractive index variation (3D: nx × ny × n_slices)
- `use_cache::Bool`: Cache shift operators (default: true)
- `trainable::Bool`: Optimize refractive index profile (default: false)
- `buffered::Bool`: Pre-allocate gradient buffers (default: false)
- `aperture::Function`: Aperture function (x, y) -> transmission (default: unity)
- `double_precision_kernel::Bool`: Use Float64 precision (default: use_cache)

# Physics

Geometric propagation:
1. Half-step shift based on tilt
2. Apply phase shift: exp(i k₀ Δn(x,y,z) Δz)
3. Full-step shift
4. Repeat for all slices

**Note:** This method ignores diffraction entirely, using only the tilt information
stored in `ScalarField`. It cannot detect phase gradients in the complex field itself.

# Use Cases

- Comparison with diffraction-based BPM to quantify diffraction effects
- Ray-tracing approximation for validation
- Tomographic reconstruction (backprojection algorithm)
- High-frequency limit where λ → 0

**Limitation:** Pure backprojection is generally inferior to diffraction-based methods
in optical regimes. Diffraction matters, especially in fiber optics and waveguide
tomography.

# Examples
```julia
# Tilted beam required
u = ScalarField(gaussian(xv, yv), (2.0, 2.0), 1.064; tilts=(0.01, 0.0))

thickness = 1000.0
n_slices = 100
dn = 0.01 * ones(256, 256, n_slices)

# Geometric shift (no diffraction)
shift_bpm = Shift_BPM(u, thickness, dn)

# Compare with diffraction
as_bpm = AS_BPM(u, thickness, 1.0, dn)

u_shift = propagate(u, shift_bpm, Forward)
u_diffraction = propagate(u, as_bpm, Forward)
# Difference shows diffraction contribution
```

See also: [`AS_BPM`](@ref), [`ShiftProp`](@ref)
"""
function Shift_BPM(u::ScalarField,
                   thickness::Real,
                   dn0::AbstractArray{<:Real};
                   use_cache::Bool = true,
                   trainable::Bool = false,
                   buffered::Bool = false,
                   aperture::Function = (_...) -> 1,
                   double_precision_kernel::Bool = use_cache)
    BPM(ShiftProp, use_cache, u, thickness, dn0; trainable, buffered, aperture,
        double_precision_kernel)
end

Functors.@functor BPM (dn,)

get_data(p::BPM) = p.dn

trainable(p::BPM{<:Trainable}) = (; dn = p.dn)

get_preallocated_gradient(p::BPM{Trainable{Buffered}}) = p.∂p

function alloc_saved_buffer(u::ScalarField, p::BPM{Trainable{Unbuffered}})
    Nv = ndims(p.dn)
    n_slices = size(p.dn, Nv)
    similar(u.electric, (size(u)..., n_slices))
end

get_saved_buffer(p::BPM{Trainable{Buffered}}) = p.u

function apply_dn_slice!(u::ScalarField,
                         dn::AbstractArray,
                         kdz,
                         direction::Type{<:Direction})
    s = sign(direction)
    lambdas = get_lambdas(u)
    @. u.electric *= cis(s*kdz/lambdas*dn)
end

function propagate!(u::ScalarField, p::BPM, direction::Type{<:Direction}; u_saved = nothing)
    Nv = ndims(p.dn)
    n_slices = size(p.dn, Nv)
    dn_slices = eachslice(p.dn, dims = Nv)
    u_saved_slices = isnothing(u_saved) ? Iterators.cycle(nothing) :
                     eachslice(u_saved, dims = ndims(u_saved))
    propagate!(u, p.p_bpm_half, direction)
    for (dn, u_saved) in zip(@view(dn_slices[1:(end - 1)]), u_saved_slices)
        copyto!(u_saved, u.electric)
        apply_dn_slice!(u, dn, p.kdz, direction)
        propagate!(u, p.p_bpm, direction)
    end
    copyto!(u_saved_slices[end], u.electric)
    apply_dn_slice!(u, dn_slices[end], p.kdz, direction)
    propagate!(u, p.p_bpm_half, direction)
    u
end

function propagate_and_save!(u::ScalarField,
                             p::BPM{Trainable{Buffered}},
                             direction::Type{<:Direction})
    propagate!(u, p, direction; u_saved = p.u)
end

function propagate_and_save!(u::ScalarField,
                             u_saved::AbstractArray,
                             p::BPM{Trainable{Unbuffered}},
                             direction::Type{<:Direction})
    propagate!(u, p, direction; u_saved)
end

function compute_dn_gradient!(∂dn::AbstractArray{T, Nd},
                              u_saved,
                              ∂u::ScalarField,
                              kdz,
                              direction) where {T <: Real, Nd}
    sdims = (Nd + 1):ndims(∂u)
    s = sign(direction)
    lambdas = get_lambdas(∂u)
    g = @. s*kdz/lambdas*imag(∂u.electric*conj(u_saved))
    copyto!(∂dn, sum(g; dims = sdims))
end

compute_dn_gradient!(::Nothing, ::Nothing, ∂u, kdz, direction) = nothing

function backpropagate!(u::ScalarField,
                        p::BPM,
                        direction::Type{<:Direction};
                        u_saved = nothing,
                        ∂p = nothing)
    Nv = ndims(p.dn)
    n_slices = size(p.dn, Nv)
    dn_slices = eachslice(p.dn, dims = Nv)
    ∂dn_slices = isnothing(∂p) ? Iterators.cycle(nothing) : eachslice(∂p.dn, dims = Nv)
    u_saved_slices = isnothing(u_saved) ? Iterators.cycle(nothing) :
                     eachslice(u_saved, dims = ndims(u_saved))
    propagate!(u, p.p_bpm_half, reverse(direction))
    for (dn, ∂dn, u_saved) in zip(@view(dn_slices[end:-1:2]),
            Iterators.reverse(∂dn_slices),
            Iterators.reverse(u_saved_slices))
        apply_dn_slice!(u, dn, p.kdz, reverse(direction))
        compute_dn_gradient!(∂dn, u_saved, u, p.kdz, direction)
        propagate!(u, p.p_bpm, reverse(direction))
    end
    apply_dn_slice!(u, dn_slices[1], p.kdz, reverse(direction))
    compute_dn_gradient!(∂dn_slices[1], u_saved_slices[1], u, p.kdz, direction)
    propagate!(u, p.p_bpm_half, reverse(direction))
    u
end

function backpropagate_with_gradient!(∂v::ScalarField,
                                      u_saved::AbstractArray,
                                      ∂p::NamedTuple,
                                      p::BPM{<:Trainable},
                                      direction::Type{<:Direction})
    ∂u = backpropagate!(∂v, p, direction; u_saved, ∂p)
    (∂u, ∂p)
end
