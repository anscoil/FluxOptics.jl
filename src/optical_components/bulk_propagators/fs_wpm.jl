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
