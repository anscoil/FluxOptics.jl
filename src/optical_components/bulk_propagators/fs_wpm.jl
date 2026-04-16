struct FS_WPM{M, A, U, T, P} <: AbstractCustomComponent{M}
    S::A
    n_slices::Int
    nz::Int
    dz::T
    n1::T
    n2::T
    k_dn_dz::T
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
        k_dn_dz = T(2π*(n2-n1)*dz) ./ get_lambdas(u)
        ∂p = (trainable && buffered) ? (; S = similar(S)) : nothing
        u_saved = (trainable && buffered) ?
            (similar(u.electric, (size(u)..., nz, 2)), similar(S, Int)) : nothing
        Us = typeof(u_saved)
        P = typeof(p_n1)
        new{M, A, Us, T, P}(S, n_slices, nz, dz, n1, n2, k_dn_dz, p_n1, p_n2, ∂p, u_saved)
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
                                         indexmap, S, k_dn_dz, ε, z, nz,
                                         ::Val{Save}) where Save
    I = @index(Global, Cartesian)
    m = smoothstep_partition(S[I], ε, z)
    phase1 = cis(k_dn_dz * (1 - m))
    phase2 = cis(-k_dn_dz * m)

    if Save
        idx = indexmap[I]
        idx += m < 1 ? 1 : 0
        indexmap[I] = idx
    end

    for J in CartesianIndices(axes(u1_e)[(ndims(S)+1):end])
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
           p.S, p.k_dn_dz, p.nz * p.dz, z, p.nz, Val(false),
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
           p.S, p.k_dn_dz, p.nz * p.dz, z, p.nz, Val(true),
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

function backpropagate!(u::ScalarField, p::FS_WPM)
    # TODO: compute u backpropagated
    u
end

function backpropagate_with_gradient!(∂v::ScalarField,
                                      u_saved::Tuple{AbstractArray, AbstractArray},
                                      ∂p::NamedTuple,
                                      p::FS_WPM{<:Trainable})
    # ∂p.S contains the gradient of p.S
    # TODO: compute ∂u 
    (∂u, ∂p)
end
