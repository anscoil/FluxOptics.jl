# HelmholtzRK4IP — Runge-Kutta in the Interaction Picture propagator for the bidirectional
# scalar Helmholtz equation.
#
# Splits the evolution operator as H = H1 + H2 where:
#   H1 = free-space propagator (exact, diagonal in Fourier space, n0 = max(n))
#   H2 = index perturbation (nilpotent, diagonal in real space)
#
# RK4 is applied to dB/dz = H̃2(z)·B in the interaction picture,
# where H̃2(z) = exp(-H1·z)·H2·exp(H1·z).
# H2 being nilpotent (H2²=0), RK4 is exact for the H2 step alone.
# Using n0 = max(n) ensures all physically propagating modes are
# represented in H1 without evanescent mode filtering.

# ─────────────────────────────────────────────────────────────────────────────
# Struct
# ─────────────────────────────────────────────────────────────────────────────

struct HelmholtzRK4IP{M, P, Q, H} <: AbstractCustomComponent{M}
    trainability::Val{M}
    prop_half :: P
    prop_full :: P
    slices :: Q
    dz :: Float64
    u_buf :: H 
    k_bufs :: NTuple{4, H}
end

Functors.@functor HelmholtzRK4IP (slices,)

function HelmholtzRK4IP(u::HelmholtzField,
                        thickness::Real,
                        n_xyz::AbstractArray{<:Number, 3};
                        n0::Number = maximum(real.(n_xyz)),
                        trainable::Bool = false,
                        buffered::Bool = false)
    n_slices = size(n_xyz, 3)
    @assert n_slices >= 1
    dz = thickness / n_slices

    prop_half = HelmholtzProp(u, dz / 2; n0)
    prop_full = HelmholtzProp(u, dz; n0)

    slices = [HelmholtzIndexSlice(u, dz, n0, n_xyz[:,:,k]; trainable, buffered)
              for k in 1:n_slices]

    # Buffers: u_buf is a full HelmholtzField clone (for u_tmp in stages 2/3/4)
    # k_bufs hold only electric_dz (electric is always zero for H2 output)
    u_buf  = similar(u)
    k_bufs = ntuple(_ -> similar(u), 4)

    M = trainability(trainable, buffered)
    HelmholtzRK4IP(Val(M), prop_half, prop_full, slices, Float64(dz), u_buf, k_bufs)
end

function _apply_H2!(k::HelmholtzField, u::HelmholtzField, slice::HelmholtzIndexSlice)
    @. k.electric = 0
    @. k.electric_dz = (2π / u.lambdas.val)^2 * (slice.n0^2 - slice.index_slice^2) * u.electric
end

function _propagate_k!(k::HelmholtzField, prop::HelmholtzProp)
    @. k.electric = 0
    propagate!(k, prop)
end

function _rk4ip_step!(u::HelmholtzField,
                      slice::HelmholtzIndexSlice,
                      p::HelmholtzRK4IP)
    dz = p.dz
    k1, k2, k3, k4 = p.k_bufs
    tmp  = p.u_buf

    # Stage 1 : k1 = H2(u) à z
    _apply_H2!(k1, u, slice)

    # Stage 2 : k2 = H2(exp(H1·dz/2)·(u + dz/2·k1)) à z+dz/2
    copyto!(tmp.electric,    u.electric)
    @. tmp.electric_dz = u.electric_dz + (dz/2) * k1.electric_dz
    propagate!(tmp, p.prop_half)
    _apply_H2!(k2, tmp, slice)

    # Stage 3 : k3 = H2(exp(H1·dz/2)·(u + dz/2·k2)) à z+dz/2
    copyto!(tmp.electric,    u.electric)
    @. tmp.electric_dz = u.electric_dz + (dz/2) * k2.electric_dz
    propagate!(tmp, p.prop_half)
    _apply_H2!(k3, tmp, slice)

    # Stage 4 : k4 = H2(exp(H1·dz)·(u + dz·k3)) à z+dz
    copyto!(tmp.electric,    u.electric)
    @. tmp.electric_dz = u.electric_dz + dz * k3.electric_dz
    propagate!(tmp, p.prop_full)
    _apply_H2!(k4, tmp, slice)

    # Combinaison : propager u et les ki vers z+dz
    propagate!(u,  p.prop_full)
    _propagate_k!(k1, p.prop_full)
    _propagate_k!(k2, p.prop_half)
    _propagate_k!(k3, p.prop_half)
    # k4 déjà à z+dz, electric=0

    @. u.electric    += (dz/6) * (k1.electric + 2*k2.electric + 2*k3.electric)
    @. u.electric_dz += (dz/6) * (k1.electric_dz + 2*k2.electric_dz +
                                   2*k3.electric_dz + k4.electric_dz)
end

function propagate!(u::HelmholtzField, p::HelmholtzRK4IP)
    for slice in p.slices
        _rk4ip_step!(u, slice, p)
    end
    u
end

propagate_and_save(p::HelmholtzRK4IP) = error("Not implemented — AD for RK4-IP pending")
