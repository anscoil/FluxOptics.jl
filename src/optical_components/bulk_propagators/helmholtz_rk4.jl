struct HelmholtzRK4{M, N, K0, K, H, T, B} <: AbstractCustomComponent{M}
    trainability::Val{M}
    n_xyz::N
    k0_sq::K0
    kxy_sq::K
    dz::Float64
    n_min_mask::T
    n_max_mask::T
    spatial_mask::B
    u_buf::H
    k_bufs::NTuple{4, H}
end

Functors.@functor HelmholtzRK4 (n_xyz,)

function HelmholtzRK4(u::HelmholtzField{U},
                      thickness::Real,
                      n_xyz::AbstractArray{<:Number, 3};
                      n_max::Real = 1.0, #maximum(real(n_xyz)),
                      trainable::Bool = false,
                      buffered::Bool = false) where {T, U <: AbstractArray{Complex{T}}}
    n_slices = size(n_xyz, 3)
    @assert n_slices >= 1
    n_min = minimum(real(n_xyz))
    dz = thickness / n_slices
    nx, ny = size(u)[1:2]
    dx, dy = u.ds
    kx = 2π .* fftfreq(nx, 1/dx)
    ky = 2π .* fftfreq(ny, 1/dy)
    kxy_sq = similar(u.electric, T, (nx, ny))
    copyto!(kxy_sq, kx.^2 .+ ky'.^2)
    k0_sq = @. (2π / u.lambdas.val)^2
    n_min_mask = @. ifelse(kxy_sq <= k0_sq * n_min^2,  one(T), zero(T))
    n_max_mask = @. ifelse(kxy_sq <= k0_sq * n_max^2,  one(T), zero(T))
    n_xyz_buf = similar(u.electric, eltype(n_xyz), size(n_xyz))
    copyto!(n_xyz_buf, n_xyz)
    spatial_mask = @. real(n_xyz_buf) < n_max
    u_buf = similar(u)
    k_bufs = ntuple(_ -> similar(u), 4)
    M = trainability(trainable, buffered)
    HelmholtzRK4(Val(M), n_xyz_buf, k0_sq, kxy_sq, Float64(dz),
                 n_min_mask, n_max_mask, spatial_mask, u_buf, k_bufs)
end

function _apply_H_full!(dv::HelmholtzField, v::HelmholtzField, p::HelmholtzRK4, k::Integer)
    n_slice = view(p.n_xyz, :, :, k)
    copyto!(dv.electric, v.electric_dz)
    copyto!(dv.electric_dz, v.electric)
    fft!(dv.electric_dz, (1,2))
    @. dv.electric_dz *= p.kxy_sq
    ifft!(dv.electric_dz, (1,2))
    # _lowpass!(dv.electric_dz, copy(v.electric), p, k)
    @. dv.electric_dz -= p.k0_sq * n_slice^2 * v.electric
    _lowpass!(dv.electric_dz, v.electric, p, k)
    # _lowpass!(dv.electric, v.electric, p, k)
end

function _lowpass!(arr::AbstractArray, arr_high::AbstractArray, p::HelmholtzRK4, k::Integer)
    fft!(arr, (1,2))
    copyto!(arr_high, arr)
    @. arr *= p.n_min_mask
    @. arr_high *= p.n_max_mask
    ifft!(arr, (1,2))
    ifft!(arr_high, (1,2))
    spatial_mask = view(p.spatial_mask, :, : , k)
    @. arr = ifelse(spatial_mask, arr, arr_high)
end

function _rk4_step!(u::HelmholtzField, p::HelmholtzRK4, k::Integer)
    dz = p.dz
    k1, k2, k3, k4 = p.k_bufs
    tmp = p.u_buf
    @. tmp.electric = u.electric
    @. tmp.electric_dz = u.electric_dz
    _apply_H_full!(k1, tmp, p, k)
    @. tmp.electric = u.electric + (dz/2) * k1.electric
    @. tmp.electric_dz = u.electric_dz + (dz/2) * k1.electric_dz
    _apply_H_full!(k2, tmp, p, k)
    @. tmp.electric = u.electric + (dz/2) * k2.electric
    @. tmp.electric_dz = u.electric_dz + (dz/2) * k2.electric_dz
    _apply_H_full!(k3, tmp, p, k)
    @. tmp.electric = u.electric + dz * k3.electric
    @. tmp.electric_dz = u.electric_dz + dz * k3.electric_dz
    _apply_H_full!(k4, tmp, p, k)
    @. u.electric += (dz/6) * (k1.electric + 2*k2.electric
                               + 2*k3.electric + k4.electric)
    @. u.electric_dz += (dz/6) * (k1.electric_dz + 2*k2.electric_dz
                                  + 2*k3.electric_dz + k4.electric_dz)
end

function propagate!(u::HelmholtzField, p::HelmholtzRK4)
    for k in 1:size(p.n_xyz, 3)
        _rk4_step!(u, p, k)
    end
    u
end

function backpropagate!(u::HelmholtzField, p::HelmholtzRK4)
    error("ask claude")
end
