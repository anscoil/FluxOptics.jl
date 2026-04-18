"""
    FS_WPM(u, ds, thickness, dz, n1, n2, f; nz=4, use_cache=true, trainable=false, buffered=false)
    FS_WPM(u, thickness, dz, n1, n2, f; kwargs...)

Freeform Surface Wave Propagation Method (FS-WPM).

Propagates scalar fields through a freeform refractive interface separating two
homogeneous media with refractive indices `n1` and `n2`. The interface is defined
by a height map `S(x,y)` and represented as a smooth transition using a smoothstep
function, enabling gradient-based shape optimization via a custom adjoint.

Goes beyond the paraxial approximation and thin-element approximation (TEA) by
accounting for the volumetric nature of the interface during propagation.

# Arguments
- `u::ScalarField`: Field template defining grid size, sampling and wavelength(s)
- `ds::NTuple`: Custom spatial sampling (defaults to `u.ds`)
- `thickness::Real`: Total thickness of the propagation volume (µm)
- `dz::Real`: Longitudinal step size (µm)
- `n1::Real`: Refractive index of the input medium
- `n2::Real`: Refractive index of the output medium
- `f`: Interface height map, either a `Function` `(x, y) -> z` or an `AbstractArray`
  (default: flat interface at z=0)
- `nz::Integer`: Number of slices stored around the interface for the adjoint (default: 4)
- `use_cache::Bool`: Cache Angular Spectrum propagation kernels (default: true)
- `trainable::Bool`: Enable gradient-based optimization of the surface `S` (default: false)
- `buffered::Bool`: Pre-allocate gradient buffers for the custom adjoint (default: false)

# Physics

Each propagation step applies:
1. Angular Spectrum propagation of `u1` in medium `n1`
2. Angular Spectrum propagation of `u2` in medium `n2`
3. Phase accumulation:
   - `u1 ← u1 · exp(i 2π Δn dz (1 - m(x,y,z)) / λ)`
   - `u2 ← u2 · exp(-i 2π Δn dz m(x,y,z) / λ)`

where `Δn = n2 - n1` and `λ` is the wavelength (broadcastable over multiple wavelengths).
4. Field stitching: `u_out = m·u1 + (1-m)·u2`

where `m(x,y,z) ∈ [0,1]` is the smoothstep partition function centered on `S(x,y)`,
`u1` carries the field as if propagating entirely in `n1`, and `u2` as if entirely
in `n2`. The stitching recombines them weighted by the local fraction of each medium.

# Examples
```julia
u = ScalarField(ones(ComplexF32, 256, 256), (2.0, 2.0), 1.55)

# Flat glass/air interface
prop = FS_WPM(u, 10.0, 0.5, 1.5, 1.0)

# Spherical freeform surface
spherical = (x, y) -> sqrt(max(0, 200^2 - x^2 - y^2)) - 200
prop_sphere = FS_WPM(u, 10.0, 0.5, 1.5, 1.0, spherical)

# Trainable surface for inverse design
prop_opt = FS_WPM(u, 10.0, 0.5, 1.5, 1.0; trainable=true, buffered=true)

# Wrap with Fourier smoothing for stable optimization
α = 1e-4
biharmonic = (kx, ky) -> 1 / (1 + 2α * (kx^2 + ky^2)^2)
wrapper = FourierSmoothingWrapper(prop_opt, (256, 256), (2.0, 2.0), biharmonic)
```

See also: [`FourierSmoothingWrapper`](@ref), [`BasisProjectionWrapper`](@ref),
[`ASProp`](@ref), [`AS_BPM`](@ref)
"""
struct FS_WPM{M, A, U, K, T, P} <: AbstractCustomComponent{M}
    S::A
    n_slices::Int
    nz::Int
    dz::T
    dn::T
    k_dz::K
    p_n1::P
    p_n2::P
    ∂p::Union{Nothing, @NamedTuple{S::A}}
    u::Union{Nothing, U}

    function FS_WPM(S::A, n_slices::Integer, nz::Integer, dz::T, dn::T, k_dz::K,
                    p_n1::P, p_n2::P, ∂p, u) where {A, T, K, P}
        M = isnothing(u) ? Trainable{Unbuffered} : Trainable{Buffered}
        U = typeof(u)
        new{M, A, U, K, T, P}(S, n_slices, nz, dz, dn, k_dz, p_n1, p_n2, ∂p, u)
    end
    
    function FS_WPM(u::ScalarField{U, Nd},
                    ds::NTuple{Nd, Real},
                    thickness::Real, dz::Real,
                    n1::Real, n2::Real,
                    f::Union{Function, AbstractArray{<:Real, Nd}} = (_...) -> 0;
                    nz::Integer = 4,
                    use_cache::Bool = true,
                    trainable::Bool = false,
                    buffered::Bool = false) where {U, Nd}
        T = real(eltype(u))
        M = trainability(trainable, buffered)
        n_slices = Int(round(thickness / dz))
        @assert Nd in (1, 2)
        @assert nz >= 0
        A = similar(U, real, Nd)
        ns = size(u)[1:Nd]
        S = isa(f, Function) ? A(function_to_array(f, ns, ds)) : A(f)
        @assert isbroadcastable(S, u)
        p_n1 = ASProp(u, dz; use_cache, n0 = n1)
        p_n2 = ASProp(u, dz; use_cache, n0 = n2)
        k_dz = T(2π*dz) ./ get_lambdas(u)
        dn = T(n1-n2)
        ∂p = (trainable && buffered) ? (; S = similar(S)) : nothing
        u_saved = (trainable && buffered) ?
            (similar(u.electric, (size(u)..., nz, 2)), similar(S, Int)) : nothing
        Us = typeof(u_saved)
        K = typeof(k_dz)
        P = typeof(p_n1)
        new{M, A, Us, K, T, P}(S, n_slices, nz, dz, dn, k_dz, p_n1, p_n2, ∂p, u_saved)
    end

    function FS_WPM(u::ScalarField{U, Nd},
                    thickness::Real, dz::Real,
                    n1::Real, n2::Real,
                    f::Union{Function, AbstractArray{<:Real, Nd}} = (_...) -> 0;
                    nz::Integer = 4,
                    use_cache::Bool = true,
                    trainable::Bool = false,
                    buffered::Bool = false) where {U, Nd}
        FS_WPM(u, Tuple(u.ds), thickness, dz, n1, n2, f; nz, use_cache, trainable, buffered)
    end
end

Functors.@functor FS_WPM (S,)

get_data(p::FS_WPM) = p.S

trainable(p::FS_WPM{<:Trainable}) = (; S = p.S)

get_preallocated_gradient(p::FS_WPM{Trainable{Buffered}}) = p.∂p

function alloc_saved_buffer(u::ScalarField, p::FS_WPM{Trainable{Unbuffered}})
    (similar(u.electric, (size(u)..., p.nz, 2)), similar(p.S, Int))
end

get_saved_buffer(p::FS_WPM{Trainable{Buffered}}) = p.u

function smoothstep(z::T) where {T}
    if z < 0
        T(0)
    elseif z < 1
        z^2*(3-2*z)
    else
        T(1)
    end
end

function smoothstep_derivative(z::T) where {T}
    if z < 0
        T(0)
    elseif z < 1
        6*z*(1-z)
    else
        T(0)
    end
end

function smoothstep_partition(sz::T, ϵ::Real, z::Real) where {T <: Real}
    smoothstep((sz - T(z))/T(ϵ) + T(0.5))
end

function smoothstep_derivative_partition(sz::T, ϵ::Real, z::Real) where {T <: Real}
    T(1/ϵ)*smoothstep_derivative((sz - T(z))/T(ϵ) + T(0.5))
end

function smoothstep_partition(p::FS_WPM; derivative::Bool = false)
    V = similar(p.S, (size(p.S)..., p.n_slices))
    zv, = spatial_vectors(p.n_slices, p.dz)
    ϵ = p.nz * p.dz
    for (z, V_slice) in zip(zv, eachslice(V; dims = ndims(V)))
        if derivative
            @. V_slice = smoothstep_derivative_partition(p.S, ϵ, z)
        else
            @. V_slice = smoothstep_partition(p.S, ϵ, z)
        end
    end
    V
end

@kernel function propagate_slice_kernel!(u1_e, u2_e, u1_s, u2_s,
                                         indexmap, S, k_dz, dn, ε, z, nz,
                                         ::Val{Save}) where Save
    I = @index(Global, Cartesian)
    m = smoothstep_partition(S[I], ε, z)

    if Save
        idx = indexmap[I]
        idx += m < 1 ? 1 : 0
        indexmap[I] = idx
    end

    for J in CartesianIndices(axes(u1_e)[(ndims(S)+1):end])
        phase1 = cis(-k_dz[J] * dn * (1 - m))
        phase2 = cis(k_dz[J] * dn * m)
        a1 = u1_e[I, J] * phase1
        a2 = u2_e[I, J] * phase2
        if Save && 1 <= idx <= nz
            u1_s[I, J, idx] = a1
            u2_s[I, J, idx] = a2
        end
        u1_e[I, J] = m * a1 + (1 - m) * a2
    end
end

function propagate_slice!(u1_e, u2_e, p, z)
    kernel = propagate_slice_kernel!(get_backend(u1_e))
    kernel(u1_e, u2_e, nothing, nothing, nothing,
           p.S, p.k_dz, p.dn, p.nz * p.dz, z, p.nz, Val(false),
           ndrange=size(p.S))
end

function propagate_slice!(u1::ScalarField, u2::ScalarField, p::FS_WPM, z::Real)
    propagate!(u1, p.p_n1)
    propagate!(u2, p.p_n2)
    u1_e = u1.electric
    u2_e = u2.electric
    propagate_slice!(u1_e, u2_e, p, z)
    u1
end

function propagate!(u::ScalarField, p::FS_WPM)
    u2 = similar(u)
    zv, = spatial_vectors(p.n_slices, p.dz)
    for z in zv
        copyto!(u2, u)
        propagate_slice!(u, u2, p, z)
    end
    u
end

function propagate_and_save_slice!(u1_e, u2_e, u1_s, u2_s, indexmap, p, z)
    kernel = propagate_slice_kernel!(get_backend(u1_e))
    kernel(u1_e, u2_e, u1_s, u2_s, indexmap,
           p.S, p.k_dz, p.dn, p.nz * p.dz, z, p.nz, Val(true),
           ndrange=size(p.S))
end

function propagate_and_save_slice!(u1::ScalarField, u2::ScalarField,
                                   u_interface::AbstractArray, indexmap::AbstractArray,
                                   p::FS_WPM, z::Real)
    propagate!(u1, p.p_n1)
    propagate!(u2, p.p_n2)
    u1_e = u1.electric
    u2_e = u2.electric
    u1_s = selectdim(u_interface, ndims(u_interface), 1)
    u2_s = selectdim(u_interface, ndims(u_interface), 2)
    propagate_and_save_slice!(u1_e, u2_e, u1_s, u2_s, indexmap, p, z)
    u1
end

function propagate_and_save!(u::ScalarField,
                             u_saved::Tuple{AbstractArray, AbstractArray},
                             p::FS_WPM{<:Trainable})
    u_interface, indexmap = u_saved
    indexmap .= 0
    u2 = similar(u)
    zv, = spatial_vectors(p.n_slices, p.dz)
    for z in zv
        copyto!(u2, u)
        propagate_and_save_slice!(u, u2, u_interface, indexmap, p, z)
    end
    u
end

@kernel function backpropagate_slice_kernel!(∂u1, ∂u2, u1, u2,
                                             indexmap, ∂S, S, k_dz, dn, ϵ, z, nz,
                                             ::Val{ComputeGrad}) where ComputeGrad
    I = @index(Global, Cartesian)
    s = S[I]
    m = smoothstep_partition(s, ϵ, z)

    if ComputeGrad
        idx = indexmap[I]
    end
    
    for J in CartesianIndices(axes(∂u1)[(ndims(S)+1):end])
        phase1 = cis(k_dz[J] * dn * (1 - m))
        phase2 = cis(-k_dz[J] * dn * m)
        ∂a = ∂u1[I, J]  # also equal to ∂u2[I, J]
        ∂a1 = ∂a * m * phase1
        ∂a2 = ∂a * (1-m) * phase2
        if ComputeGrad && 1 <= idx <= nz
            a1 = u1[I, J, idx]
            a2 = u2[I, J, idx]
            ∂m1 = real(conj(a1 - a2) * ∂a)
            ∂m2 = k_dz[J] * dn * imag(conj(a1)*∂a1 + conj(a2)*∂a2)
            Dm = smoothstep_derivative_partition(s, ϵ, z)
            ∂s = ∂S[I]
            ∂S[I] = ∂s + Dm * (∂m1 + ∂m2)
        end
        ∂u1[I, J] = ∂a1
        ∂u2[I, J] = ∂a2
    end

    if ComputeGrad
        idx -= m < 1 ? 1 : 0
        indexmap[I] = idx
    end
end

function backpropagate_slice!(∂u1, ∂u2, p, z)
    kernel = backpropagate_slice_kernel!(get_backend(∂u1))
    kernel(∂u1, ∂u2, nothing, nothing, nothing,
           nothing, p.S, p.k_dz, p.dn, p.nz * p.dz, z, p.nz, Val(false),
           ndrange=size(p.S))
end

function backpropagate_slice!(u1::ScalarField, u2::ScalarField, p::FS_WPM, z::Real)
    u1_e = u1.electric
    u2_e = u2.electric
    backpropagate_slice!(u1_e, u2_e, p, z)
    backpropagate!(u1, p.p_n1)
    backpropagate!(u2, p.p_n2)
    @. u1_e = u1_e + u2_e
    u1
end

function backpropagate!(u::ScalarField, p::FS_WPM)
    u2 = similar(u)
    zv, = spatial_vectors(p.n_slices, p.dz)
    for z in reverse(zv)
        copyto!(u2, u)
        backpropagate_slice!(u, u2, p, z)
    end
    u
end

function backpropagate_with_gradient_slice!(∂S, ∂u1, ∂u2, u1, u2, indexmap, p, z)
    kernel = backpropagate_slice_kernel!(get_backend(∂u1))
    kernel(∂u1, ∂u2, u1, u2, indexmap,
           ∂S, p.S, p.k_dz, p.dn, p.nz * p.dz, z, p.nz, Val(true),
           ndrange=size(p.S))
end

function backpropagate_with_gradient_slice!(∂S::AbstractArray,
                                            u1::ScalarField, u2::ScalarField,
                                            u_interface::AbstractArray,
                                            indexmap::AbstractArray,
                                            p::FS_WPM, z::Real)
    u1_e = u1.electric
    u2_e = u2.electric
    u1_s = selectdim(u_interface, ndims(u_interface), 1)
    u2_s = selectdim(u_interface, ndims(u_interface), 2)
    backpropagate_with_gradient_slice!(∂S, u1_e, u2_e, u1_s, u2_s, indexmap, p, z)
    backpropagate!(u1, p.p_n1)
    backpropagate!(u2, p.p_n2)
    @. u1_e = u1_e + u2_e
    u1, ∂S
end

function backpropagate_with_gradient!(∂v::ScalarField,
                                      u_saved::Tuple{AbstractArray, AbstractArray},
                                      ∂p::NamedTuple,
                                      p::FS_WPM{<:Trainable})
    u_interface, indexmap = u_saved
    ∂p.S .= 0
    ∂v2 = similar(∂v)
    zv, = spatial_vectors(p.n_slices, p.dz)
    for z in reverse(zv)
        copyto!(∂v2, ∂v)
        backpropagate_with_gradient_slice!(∂p.S, ∂v, ∂v2, u_interface, indexmap, p, z)
    end
    (∂v, ∂p)
end
