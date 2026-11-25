# ---------------------------------------------------------------------------------------
# Propagation factors
#
# A pass carries the modes departing in its direction over z, with the factors exp(i·kz·z)
# of propagation_factors. A component with a state needs nothing else: the modes
# travelling against the pass are read from its state. A stateless component also
# transports these opposite modes across z, with the factors of transport_factors.
#
# Media symmetric in z (backward eigenvalues = -forward ones) have the same factor in both
# directions (SymmetricFactors); general media have one per direction (PropagationFactors).
# ---------------------------------------------------------------------------------------

"""Factors of a medium symmetric in z: the same for both directions."""
struct SymmetricFactors{V}
    factor::V
end

"""Factors of a general medium: fwd for forward passes, bwd for backward passes."""
struct PropagationFactors{V}
    fwd::V
    bwd::V
end

const ModalFactors = Union{SymmetricFactors, PropagationFactors}

pass_factor(f::SymmetricFactors, ::Direction) = f.factor
pass_factor(f::PropagationFactors, ::Forward) = f.fwd
pass_factor(f::PropagationFactors, ::Backward) = f.bwd

# Departing modes: exp(i·kz_f·z) for the forward modes on a forward pass, exp(-i·kz_b·z)
# for the backward modes on a backward pass
function propagation_factors(m::AbstractModeBasis, z::Real)
    kz = eigenvalues(m)
    iz = im * real(eltype(kz))(z)
    PropagationFactors(exp.(iz .* kz[mode_indices(m, Forward())]),
                       exp.(-iz .* kz[mode_indices(m, Backward())]))
end

function propagation_factors(m::ScalarModeBasis, z::Real)
    SymmetricFactors(exp(im * real(typeof(m.kz))(z) * m.kz))
end

function propagation_factors(m::IsotropicModeBasis, z::Real)
    SymmetricFactors(exp(im * real(typeof(m.kz))(z) * m.kz))
end

# Transport of a mode of eigenvalue kz over the signed distance Δz: exp(i·kz·Δz). Exact
# for a propagating mode; for an evanescent mode (Re(kz²) < 0) the transport against its
# decay grows as exp(κ|Δz|), so it is cut unless keep_evanescent.
function transport(kz, Δz, keep_evanescent::Bool)
    keep_evanescent || real(kz^2) >= 0 ? exp(im * Δz * kz) : zero(kz)
end

# Opposite modes of stateless passes: the backward modes over +z on a forward pass, the
# forward modes over -z on a backward pass
function transport_factors(m::AbstractModeBasis, z::Real; keep_evanescent::Bool = false)
    kz = eigenvalues(m)
    Z = real(eltype(kz))(z)
    t(k, Δz) = transport(k, Δz, keep_evanescent)
    PropagationFactors(t.(kz[mode_indices(m, Backward())], Z),
                       t.(kz[mode_indices(m, Forward())], -Z))
end

function transport_factors(m::ScalarModeBasis, z::Real; keep_evanescent::Bool = false)
    SymmetricFactors(transport(m.kz, -real(typeof(m.kz))(z), keep_evanescent))
end

function transport_factors(m::IsotropicModeBasis, z::Real; keep_evanescent::Bool = false)
    SymmetricFactors(transport(m.kz, -real(typeof(m.kz))(z), keep_evanescent))
end

# Filter of the orders |k⊥| > kmax (kmax = nothing: no filter). The modal factors of the
# filtered orders are set to zero, so that every modal step removes them: what a kick
# puts there does not reach the next kick. With a reference medium in which an object of
# higher index is evanescent for some orders (k_ref < |k⊥| < k_object), the kicks amplify
# them slice after slice unless the whole fixed point is solved (truncations, Neumann
# iterations); orders near the cutoff of a periodic window also make it resonate. kmax
# below k_ref removes both, at the price of the light that the object sends there.
#
# knotch = (k1, k2) removes instead the band k1 < |k⊥| < k2 only: around the cutoff k_ref
# of the reference medium, the orders with |kz| small, which make a periodic window
# resonate (kicks weighted by 1/kz), while the evanescent orders beyond k2, where an object
# of higher index carries light, are kept. Both filters can be combined.
function order_mask(u::AbstractArray, ns::NTuple{2, Integer}, ds, kmax, knotch = nothing)
    isnothing(kmax) && isnothing(knotch) && return nothing
    T = real(eltype(u))
    fx = T(2π) .* fftfreq(ns[1], 1 / ds[1])
    fy = T(2π) .* fftfreq(ns[2], 1 / ds[2])
    k2max = isnothing(kmax) ? T(Inf) : T(kmax)^2
    k1², k2² = isnothing(knotch) ? (T(Inf), T(Inf)) : (T(knotch[1])^2, T(knotch[2])^2)
    keep = similar(u, Bool, ns)
    copyto!(keep, [(a^2 + b^2 <= k2max) && !(k1² < a^2 + b^2 < k2²) for a in fx, b in fy])
    keep
end

cut_orders(factors, ::Nothing) = factors
cut_orders(::Nothing, keep) = nothing
cut_orders(::Nothing, ::Nothing) = nothing

function cut_orders(factors::StructArray, keep::AbstractArray{Bool})
    cut(c) = ifelse.(keep, c, Ref(zero(eltype(c))))
    StructArray{eltype(factors)}(map(cut, StructArrays.components(factors)))
end

# Coefficients of the modal kernels: the transport factors only without a state
modal_coefs(medium, factors, ::Nothing) = (medium.modes, factors)
modal_coefs(medium, factors, transports) = (medium.modes, factors, transports)

# ---------------------------------------------------------------------------------------
# Modal step
# ---------------------------------------------------------------------------------------

# With a state: the opposite amplitudes are read from it, the departing ones stored
function propagate_modes(m::AbstractModeBasis, f::ModalFactors,
                         direction::Direction, ψ::Tuple, stored::Tuple)
    a = decompose(m, SVector(ψ), direction)
    d = pass_factor(f, direction) .* a
    Ψ = recombine(m, d, SVector(stored), direction)
    # Ψ = recompose(m, d, direction) + recompose(m, SVector(stored), reverse(direction))
    Tuple(Ψ), Tuple(a)
end

# Stateless: the opposite modes are transported across z
function propagate_modes(m::AbstractModeBasis, f::ModalFactors, t::ModalFactors,
                         direction::Direction, ψ::Tuple, ::Tuple{})
    Ψ = SVector(ψ)
    d = pass_factor(f, direction) .* decompose(m, Ψ, direction)
    o = pass_factor(t, direction) .* decompose(m, Ψ, reverse(direction))
    # Tuple(recompose(m, d, direction) + recompose(m, o, reverse(direction))), ()
    Tuple(recombine(m, d, o, direction)), ()
end

function propagate_modes_adjoint(m::AbstractModeBasis, f::ModalFactors,
                                 direction::Direction, ∂ψ::Tuple, ∂stored::Tuple)
    ∂Ψ = SVector(∂ψ)
    ∂d = conj.(pass_factor(f, direction)) .* recompose_adjoint(m, ∂Ψ, direction)
    ∂o = recompose_adjoint(m, ∂Ψ, reverse(direction))
    Tuple(decompose_adjoint(m, ∂d + SVector(∂stored), direction)), Tuple(∂o)
end

function propagate_modes_adjoint(m::AbstractModeBasis, f::ModalFactors, t::ModalFactors,
                                 direction::Direction, ∂ψ::Tuple, ::Tuple{})
    ∂Ψ = SVector(∂ψ)
    ∂d = conj.(pass_factor(f, direction)) .* recompose_adjoint(m, ∂Ψ, direction)
    ∂o = conj.(pass_factor(t, direction)) .* recompose_adjoint(m, ∂Ψ, reverse(direction))
    Tuple(decompose_adjoint(m, ∂d, direction) +
          decompose_adjoint(m, ∂o, reverse(direction))), ()
end

@kernel function modal_kernel!(f, fields, stored, coefs, direction)
    I = @index(Global, Cartesian)
    c = map(a -> a[I], coefs)
    for J in CartesianIndices(axes(first(fields))[3:end])
        ψ = map(a -> a[I, J], fields)
        a_st = map(a -> a[I, J], stored)
        ψ_out, a_out = f(c..., direction, ψ, a_st)
        map((a, v) -> a[I, J] = v, fields, ψ_out)
        map((a, v) -> a[I, J] = v, stored, a_out)
    end
end

function launch_modal!(f, fields::Tuple, stored::Tuple, coefs::Tuple, direction::Direction)
    if linear_host((fields..., stored...))       # see launch_combine!
        modal_host!(f, fields, stored, coefs, direction)
    else
        modal_kernel!(get_backend(first(fields)))(
            f, fields, stored, coefs, direction; ndrange = size(first(fields))[1:2])
    end
end

# Fields of shape (Nx, Ny, batch...): the outer loop runs over the 2D slices of the
# batch, the inner loop over their pixels with one linear index, so @simd applies.
# Coefficients are (Nx, Ny); fields and stored states share the shape of the first field.
function modal_host!(f, fields::Tuple, stored::Tuple, coefs::Tuple, direction::Direction)
    A = first(fields)
    n = size(A, 1) * size(A, 2)                  # pixels of one slice
    for o in 0:n:(length(A) - n)                 # offset of each batch slice
        @inbounds @simd for i in 1:n
            c = map(a -> @inbounds(a[i]), coefs)
            ψ_out, a_out = @inline f(c..., direction, map(a -> @inbounds(a[o + i]), fields),
                                     map(a -> @inbounds(a[o + i]), stored))
            map((a, v) -> @inbounds(a[o + i] = v), fields, ψ_out)
            map((a, v) -> @inbounds(a[o + i] = v), stored, a_out)
        end
    end
end

include("scalar_wave_propagator.jl")
include("vectorial_propagator.jl")
