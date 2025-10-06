"""
    TeaDOE(u::ScalarField, Δn, f; trainable=false, buffered=false)

Create a diffractive optical element using thin element approximation.

Models a phase element with refractive index difference Δn and surface height h(x,y).
The transmission is: `t = exp(i 2π Δn h(x,y) / λ)`, wavelength-dependent.

# Arguments
- `u::ScalarField`: Field template
- `Δn`: Refractive index difference (n_element - n_surround)
- `f`: Height function `(x, y) -> h` or height array
- `trainable::Bool`: Optimize surface profile (default: false)
- `buffered::Bool`: Pre-allocate gradients (default: false)

# Examples
```julia
u = ScalarField(ones(ComplexF64, 256, 256), (1.0, 1.0), 1.064)

# Sinusoidal grating
Δn = 0.5
grating = TeaDOE(u, Δn, (x, y) -> 0.5 * sin(2π * x / 50))

# Trainable DOE for beam shaping
doe_opt = TeaDOE(u, 0.5, (x, y) -> 0.0; trainable=true)

# Multi-level DOE
levels = compute_multilevel_profile(...)
doe_ml = TeaDOE(u, 0.5, levels)
```

See also: [`TeaReflector`](@ref), [`Phase`](@ref)
"""
struct TeaDOE{M, Fn, Fr, A, U} <: AbstractCustomComponent{M}
    dn::Fn
    r::Fr
    h::A
    ∂p::Union{Nothing, @NamedTuple{h::A}}
    u::Union{Nothing, U}

    function TeaDOE(dn::Fn,
                    r::Fr,
                    h::A,
                    ∂p::Union{Nothing, @NamedTuple{h::A}},
                    u::U) where {Fn, Fr, A, U}
        M = isnothing(u) ? Trainable{Unbuffered} : Trainable{Buffered}
        new{M, Fn, Fr, A, U}(dn, r, h, ∂p, u)
    end

    function TeaDOE(u::ScalarField{U, Nd},
                    ds::NTuple{Nd, Real},
                    dn::Union{Real, Function},
                    f::Function = (_...) -> 0;
                    r::Union{Number, Function} = 1,
                    trainable::Bool = false,
                    buffered::Bool = false) where {N, Nd, T,
                                                   U <: AbstractArray{Complex{T}, N}}
        @assert Nd in (1, 2)
        @assert N >= Nd
        M = trainability(trainable, buffered)
        P = similar(U, real, Nd)
        ns = size(u)[1:Nd]
        h = P(function_to_array(f, ns, ds))
        ∂p = (trainable && buffered) ? (; h = similar(h)) : nothing
        u = (trainable && buffered) ? similar(u.electric) : nothing
        dn_f = isa(dn, Real) ? (λ -> T(dn)) : (λ -> T(dn(λ)))
        r_f = isa(r, Number) ? (λ -> Complex{T}(r)) : (λ -> Complex{T}(r(λ)))
        Fn = typeof(dn_f)
        Fr = typeof(r_f)
        A = typeof(h)
        new{M, Fn, Fr, A, U}(dn_f, r_f, h, ∂p, u)
    end

    function TeaDOE(u::ScalarField{U, Nd},
                    dn::Union{Real, Function},
                    f::Function = (_...) -> 0;
                    r::Union{Number, Function} = 1,
                    trainable::Bool = false,
                    buffered::Bool = false) where {U <: AbstractArray{<:Complex}, Nd}
        TeaDOE(u, Tuple(u.ds), dn, f; r, trainable, buffered)
    end
end

"""
    TeaReflector(u::ScalarField, f; r=(λ)->1.0, trainable=false, buffered=false)

Create a reflective diffractive element with variable surface height.

Models a mirror with surface profile h(x,y) and optional wavelength-dependent
reflectivity r(λ). Phase shift: `φ = 4π h(x,y) / λ` (factor 2 from reflection).

# Arguments
- `u::ScalarField`: Field template
- `f`: Height function `(x, y) -> h` or height array
- `r`: Reflectivity function `(λ) -> r` (complex, default: 1.0)
- `trainable::Bool`: Optimize surface (default: false)
- `buffered::Bool`: Pre-allocate gradients (default: false)

# Examples
```julia
u = ScalarField(ones(ComplexF64, 256, 256), (1.0, 1.0), 1.064)

# Simple mirror with surface
mirror = TeaReflector(u, (x, y) -> 0.01 * (x^2 + y^2))

# Mirror with wavelength-dependent coating
r_coating = λ -> 0.95 * exp(im * π/4)  # 95% reflective + phase
mirror_coated = TeaReflector(u, (x, y) -> 0.0; r=r_coating)

# Deformable mirror (trainable)
dm = TeaReflector(u, (x, y) -> 0.0; trainable=true)
```

See also: [`TeaDOE`](@ref), [`Phase`](@ref)
"""
function TeaReflector(u::ScalarField{U, Nd},
                      ds::NTuple{Nd, Real},
                      f::Function = (_...) -> 0;
                      r::Union{Number, Function} = 1,
                      trainable::Bool = false,
                      buffered::Bool = false) where {U <: AbstractArray{<:Complex}, Nd}
    TeaDOE(u, ds, 2, f; r, trainable, buffered)
end

function TeaReflector(u::ScalarField{U, Nd},
                      f::Function = (_...) -> 0;
                      r::Union{Number, Function} = 1,
                      trainable::Bool = false,
                      buffered::Bool = false) where {U <: AbstractArray{<:Complex}, Nd}
    TeaDOE(u, 2, f; r, trainable, buffered)
end

Functors.@functor TeaDOE (h,)

get_data(p::TeaDOE) = p.h

trainable(p::TeaDOE{<:Trainable}) = (; h = p.h)

get_preallocated_gradient(p::TeaDOE{Trainable{Buffered}}) = p.∂p

alloc_saved_buffer(u::ScalarField, p::TeaDOE{Trainable{Unbuffered}}) = similar(u.electric)

get_saved_buffer(p::TeaDOE{Trainable{Buffered}}) = p.u

function apply_phase!(u::AbstractArray{T}, lambdas, p::TeaDOE, ::Type{Forward}) where {T}
    @. u *= p.r(lambdas) * cis((T(2)*π/lambdas)*p.dn(lambdas)*p.h)
end

function apply_phase!(u::AbstractArray{T}, lambdas, p::TeaDOE, ::Type{Backward}) where {T}
    @. u *= conj(p.r(lambdas)) * cis(-(T(2)*π/lambdas)*p.dn(lambdas)*p.h)
end

function propagate!(u::ScalarField, p::TeaDOE, direction::Type{<:Direction})
    apply_phase!(u.electric, get_lambdas(u), p, direction)
    u
end

function backpropagate!(u::ScalarField, p::TeaDOE, direction::Type{<:Direction})
    propagate!(u, p, reverse(direction))
end

function propagate_and_save!(u::ScalarField,
                             p::TeaDOE{Trainable{Buffered}},
                             direction::Type{<:Direction})
    copyto!(p.u, u.electric)
    propagate!(u, p, direction)
end

function propagate_and_save!(u::ScalarField,
                             u_saved,
                             p::TeaDOE{Trainable{Unbuffered}},
                             direction::Type{<:Direction})
    copyto!(u_saved, u.electric)
    propagate!(u, p, direction)
end

function compute_surface_gradient!(∂h::P,
                                   u_saved,
                                   ∂u::ScalarField,
                                   dn,
                                   r,
                                   direction) where {T <: Real, Nd,
                                                     P <: AbstractArray{T, Nd}}
    sdims = (Nd + 1):ndims(∂u)
    s = sign(direction)
    lambdas = get_lambdas(∂u)
    g = @. (s*T(2)*π*dn(lambdas)/lambdas)*imag(∂u.electric*conj(u_saved))
    copyto!(∂h, sum(g; dims = sdims))
end

function backpropagate_with_gradient!(∂v::ScalarField,
                                      u_saved::AbstractArray,
                                      ∂p::NamedTuple,
                                      p::TeaDOE{<:Trainable},
                                      direction::Type{<:Direction})
    ∂u = backpropagate!(∂v, p, direction)
    compute_surface_gradient!(∂p.h, u_saved, ∂u, p.dn, p.r, direction)
    (∂u, ∂p)
end
