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
        A = similar(U, real, Nd)
        ns = size(u)[1:Nd]
        S = isa(f, Function) ? A(function_to_array(f, ns, ds)) : A(f)
        p_n1 = ASProp(u, dz; use_cache, n0 = n1)
        p_n2 = ASProp(u, dz; use_cache, n0 = n2)
        k_dn_dz = T(2π*(n2-n1)*dz) ./ get_lambdas(u)
        ∂p = (trainable && buffered) ? (; S = similar(S)) : nothing
        u_sav = (trainable && buffered) ? similar(u.electric, (size(u)..., nz, 2)) : nothing
        P = typeof(p_n1)
        new{M, A, U, T, P}(S, n_slices, nz, dz, n1, n2, k_dn_dz, p_n1, p_n2, ∂p, u_sav)
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

function smoothstep(z::T) where {T}
    if z < 0
        T(0)
    elseif z < 1
        z^2*(3-2*z)
    else
        T(1)
    end
end

function smoothstep_derivative(z::Real)
    if z < 0
        T(0)
    elseif z < 1
        6*z*(1-z)
    else
        T(1)
    end
end

function smoothstep_partition(sz::T, ϵ::Real, z::Real) where {T <: Real}
    smoothstep((sz - T(z))/T(ϵ) + 0.5)
end

function smoothstep_partition(p::FS_WPM)
    V = similar(p.S, (size(p.S)..., p.n_slices))
    zv, = spatial_vectors(p.n_slices, p.dz)
    ϵ = p.nz * p.dz
    for (z, V_slice) in zip(zv, eachslice(V; dims = ndims(V)))
        @. V_slice = smoothstep_partition(p.S, ϵ, z)
    end
    V
end
