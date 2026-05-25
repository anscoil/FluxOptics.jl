struct BidirectionalKernel{K, T}
    a::K
    r01::K
    t10::K
    r02::K
    exp_a_p::K
    exp_a_m::K
    nrm_f::T
end

Adapt.@adapt_structure BidirectionalKernel

struct BidirectionalBPM{M, N, K, E, U, T, P}  <: AbstractCustomComponent{M}
    trainability::Val{M}
    n_xyz::N
    n0::Complex{T}
    n1::Complex{T}
    n2::Complex{T}
    dz::T
    p_f::P
    E_tmp::E
    ub::HelmholtzField{U}
    β::Ref{T}
    kernel::BidirectionalKernel{K, T}
end

Functors.@functor BidirectionalBPM (n_xyz,)

function optimal_gauge(n_xyz, fill_factor=0.5)
    re_n2 = real.(n_xyz.^2)
    n0_real = sqrt(fill_factor * maximum(re_n2) + (1-fill_factor) * minimum(re_n2))
    n0_imag = 0.05 * n0_real
    return complex(n0_real, n0_imag)
end

function BidirectionalBPM(u::HelmholtzField{U},
                          thickness::Real,
                          n_xyz::AbstractArray{<:Number, 3};
                          n0::Number = optimal_gauge(n_xyz),
                          n1::Number = 1.0,
                          n2::Number = 1.0,
                          trainable::Bool = false,
                          buffered::Bool = false) where {T <: Real,
                                                         U <: AbstractArray{Complex{T}}}
    ns = size(u)[1:2]
    n_slices = size(n_xyz, 3)
    @assert size(n_xyz)[1:2] == ns
    @assert n_slices >= 2
    dz = T(thickness / n_slices)
    M  = trainability(trainable, buffered)
    N = isreal(n_xyz) ? T : Complex{T}
    n_xyz_buf = similar(u.electric, N, size(n_xyz))
    copyto!(n_xyz_buf, n_xyz)
    n0 = Complex{T}(n0)
    n1 = Complex{T}(n1)
    n2 = Complex{T}(n2)
    a = im * compute_kz(u, n0)
    r01 = compute_fresnel_r12(u, n0, n1)
    t10 = compute_fresnel_t12(u, n1, n0)
    r02 = compute_fresnel_r12(u, n0, n2)
    exp_a_p = @. exp(a * dz)
    exp_a_m = @. exp(-a * dz)
    u_plan = similar(u.electric)
    p_f = make_fft_plans(u_plan, (1, 2); normalize = false)
    E_tmp = similar(u.electric, (size(u.electric)..., n_slices))
    @. E_tmp = 0
    ub = similar(u)
    @. ub.electric = 0
    @. ub.electric_dz = 0
    nrm_f = T(1/prod(ns))
    kernel = BidirectionalKernel(a, r01, t10, r02, exp_a_p, exp_a_m, nrm_f)
    BidirectionalBPM(Val(M), n_xyz_buf, n0, n1, n2, dz, p_f, E_tmp, ub, Ref(T(0)), kernel)
end

_get_val(A::AbstractArray{<:Any, 2}, I, J) = A[I]
function _get_val(A::AbstractArray{<:Any, N}, I, J) where {N}
    A[I, CartesianIndex(min.(Tuple(J), size(A)[3:end]))]
end

@kernel function propagate_helmholtz_forward_kernel!(electric, electric_dz, E_tmp, kernel, β)
    nrm_f = kernel.nrm_f
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(electric)[3:end])
        a = _get_val(kernel.a, I, J)
        exp_a_p = _get_val(kernel.exp_a_p, I, J)
        exp_a_m = _get_val(kernel.exp_a_m, I, J)
        E_val = electric[I, J]
        dE_val = electric_dz[I, J]
        E_minus = (1-β) * E_tmp[I, J]
        if β > 0
            E_minus += 0.5 * β * (E_val - dE_val / a)
        end
        E_plus = 0.5 * (E_val + dE_val / a)
        E_plus *= exp_a_p
        E_minus *= exp_a_m
        electric[I,J] = nrm_f * (E_plus + E_minus)
        electric_dz[I,J] = nrm_f * a * (E_plus - E_minus)
        E_tmp[I, J] = E_plus
    end
end

@kernel function propagate_helmholtz_backward_kernel!(electric, electric_dz, E_tmp, kernel, β)
    nrm_f = kernel.nrm_f
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(electric)[3:end])
        a = _get_val(kernel.a, I, J)
        exp_a_p = _get_val(kernel.exp_a_p, I, J)
        exp_a_m = _get_val(kernel.exp_a_m, I, J)
        E_val = electric[I, J]
        dE_val = electric_dz[I, J]
        E_plus = (1-β) * E_tmp[I, J]
        if β > 0
            E_plus += 0.5 * β * (E_val + dE_val / a)
        end
        E_minus = 0.5 * (E_val - dE_val / a)
        E_plus *= exp_a_m
        E_minus *= exp_a_p
        electric[I,J] = nrm_f * (E_plus + E_minus)
        electric_dz[I,J] = nrm_f * a * (E_plus - E_minus)
        E_tmp[I, J] = E_minus
    end
end

@kernel function propagate_helmholtz_kernel!(electric, electric_dz, E_tmp,
                                             kernel, β, ::Val{forward}) where {forward}
    nrm_f = kernel.nrm_f
    s = forward ? 1 : -1
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(electric)[3:end])
        a = _get_val(kernel.a, I, J)
        exp_a_p = _get_val(kernel.exp_a_p, I, J)
        exp_a_m = _get_val(kernel.exp_a_m, I, J)
        E_val = electric[I, J]
        dE_val = electric_dz[I, J]
        E2 = (1-β) * E_tmp[I, J]
        if β > 0
            E2 += 0.5 * β * (E_val - s * dE_val / a)
        end
        E1 = 0.5 * (E_val + s * dE_val / a)
        E1 *= exp_a_p
        E2 *= exp_a_m # conj(exp_a_p)
        electric[I,J] = nrm_f * (E1 + E2)
        electric_dz[I,J] = nrm_f * a * s * (E1 - E2)
        E_tmp[I,J] = E1
    end
end

function propagate_slice_forward!(u::HelmholtzField, p::BidirectionalBPM, k::Integer)
    β = p.β[]
    @assert 0 <= β <= 1
    backend = get_backend(u.electric)
    n_xy = view(p.n_xyz, :, :, k)
    @. u.electric_dz += ((2π/u.lambdas.val)^2 * (p.n0^2 - n_xy^2) * p.dz * u.electric)
    compute_ft!(p.p_f, u)
    E_minus = selectdim(p.E_tmp, ndims(p.E_tmp), k)
    propagate_helmholtz_kernel!(backend)(
        u.electric, u.electric_dz, E_minus, p.kernel, β, Val(true);
        ndrange = size(u.electric)[1:2])
    compute_ift!(p.p_f, u)
end

function propagate_slice_backward!(u::HelmholtzField, p::BidirectionalBPM, k::Integer)
    β = p.β[]
    @assert 0 <= β <= 1
    backend = get_backend(u.electric)
    n_xy = view(p.n_xyz, :, :, k)
    compute_ft!(p.p_f, u)
    E_plus = selectdim(p.E_tmp, ndims(p.E_tmp), k)
    propagate_helmholtz_kernel!(backend)(
        u.electric, u.electric_dz, E_plus, p.kernel, β, Val(false);
        ndrange = size(u.electric)[1:2])
    compute_ift!(p.p_f, u)
    @. u.electric_dz -= ((2π/u.lambdas.val)^2 * (p.n0^2 - n_xy^2) * p.dz * u.electric)
end

@kernel function boundary_condition_kernel_1!(electric, electric_dz, E_f, kernel)
    nrm_f = kernel.nrm_f
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(electric)[3:end])
        a = _get_val(kernel.a, I, J)
        r01 = _get_val(kernel.r01, I, J)
        t10 = _get_val(kernel.t10, I, J)
        E_val = electric[I, J]
        dE_val = electric_dz[I, J]
        E_plus_ref = E_f[I, J]
        E_minus = 0.5 * (E_val - dE_val / a)
        E_plus = t10 * E_plus_ref + r01 * E_minus
        electric[I,J] = nrm_f * (E_plus + E_minus)
        electric_dz[I,J] = nrm_f * a * (E_plus - E_minus)
    end
end

function apply_boundary_condition_1!(u::HelmholtzField,
                                     E_f::AbstractArray,
                                     p::BidirectionalBPM)
    backend = get_backend(u.electric)
    compute_ft!(p.p_f, u)
    boundary_condition_kernel_1!(backend)(
        u.electric, u.electric_dz, E_f, p.kernel; ndrange = size(u.electric)[1:2])
    compute_ift!(p.p_f, u)
end

@kernel function boundary_condition_kernel_2!(electric, electric_dz, kernel)
    nrm_f = kernel.nrm_f
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(electric)[3:end])
        a = _get_val(kernel.a, I, J)
        r02 = _get_val(kernel.r02, I, J)
        E_val = electric[I, J]
        dE_val = electric_dz[I, J]
        E_plus = 0.5 * (E_val + dE_val / a)
        E_minus = r02 * E_plus
        electric[I,J] = nrm_f * (E_plus + E_minus)
        electric_dz[I,J] = nrm_f * a * (E_plus - E_minus)
    end
end

function apply_boundary_condition_2!(u::HelmholtzField, p::BidirectionalBPM)
    backend = get_backend(u.electric)
    compute_ft!(p.p_f, u)
    boundary_condition_kernel_2!(backend)(
        u.electric, u.electric_dz, p.kernel; ndrange = size(u.electric)[1:2])
    compute_ift!(p.p_f, u)
end

function _propagate!(u::HelmholtzField, p::BidirectionalBPM)
    slices_range = 1:size(p.n_xyz, 3)
    for k in reverse(slices_range)
        propagate_slice_backward!(p.ub, p, k)
    end
    kz = compute_kz(u, p.n1)
    E_f = fft(u.electric, (1, 2))
    dEdz_f = fft(u.electric_dz, (1, 2))
    @. E_f = 0.5 * (E_f + dEdz_f  / (im * kz))
    copyto!(u, p.ub)
    apply_boundary_condition_1!(u, E_f, p)
    for k in slices_range
        propagate_slice_forward!(u, p, k)
    end
    apply_boundary_condition_2!(u, p)
    copyto!(p.ub, u)
    u
end

function pack_state!(x::AbstractVector, p::BidirectionalBPM)
    ne = length(p.E_tmp)
    nu = length(p.ub.electric)
    copyto!(view(x, 1:ne), vec(p.E_tmp))
    copyto!(view(x, ne+1:ne+nu), vec(p.ub.electric))
    copyto!(view(x, ne+nu+1:length(x)), vec(p.ub.electric_dz))
    x
end

function unpack_state!(p::BidirectionalBPM, x::AbstractVector)
    ne = length(p.E_tmp)
    nu = length(p.ub.electric)
    copyto!(vec(p.E_tmp), view(x, 1:ne))
    copyto!(vec(p.ub.electric), view(x, ne+1:ne+nu))
    copyto!(vec(p.ub.electric_dz), view(x, ne+nu+1:length(x)))
    p
end

function andersonm!(x::AbstractVector, F!; m::Int=3, maxiter=50, tol=1f-6)
    T = real(eltype(x))
    G = [similar(x) for _ in 1:m]
    Fx = [similar(x) for _ in 1:m]
    x_prev = similar(x)
    x_best = similar(x)
    g_cur = similar(x)
    slot = 0
    n_fill = 0
    res_best = T(Inf)

    for iter in 1:maxiter
        copyto!(x_prev, x)
        F!(x)
        @. g_cur = x - x_prev

        res = T(norm(g_cur))
        # @info "iter $iter, res = $res"
        if res < res_best
            res_best = res
            copyto!(x_best, x)
        end
        res < tol && break

        if res > 10 * res_best          # rollback + restart
            copyto!(x, x_best)
            slot = 0
            n_fill = 0
            continue
        end

        slot = mod1(slot + 1, m)
        n_fill = min(n_fill + 1, m)
        copyto!(G[slot], g_cur)
        copyto!(Fx[slot], x)

        n_fill < 2 && continue

        k = n_fill
        idx = [mod1(slot - k + i, m) for i in 1:k]

        H = [real(dot(G[idx[i]], G[idx[j]])) for i in 1:k, j in 1:k]

        A  = [H              ones(T, k, 1);
              ones(T, 1, k)  zeros(T, 1, 1)]
        b  = vcat(zeros(T, k), T(1))
        θ  = (A \ b)[1:k]

        fill!(x, zero(eltype(x)))
        for i in 1:k
            @. x += θ[i] * Fx[idx[i]]
        end
    end
    x
end

function propagate_anderson!(u::HelmholtzField, p::BidirectionalBPM, anderson!;
                             maxiter=5, tol=1f-6)
    u_inc_e = copy(u.electric)
    u_inc_edz = copy(u.electric_dz)

    x = similar(p.E_tmp, eltype(p.E_tmp),
                length(p.E_tmp) + 2 * length(p.ub.electric))
    pack_state!(x, p)

    function F!(x)
        unpack_state!(p, x)
        copyto!(u.electric, u_inc_e)
        copyto!(u.electric_dz, u_inc_edz)
        _propagate!(u, p)
        pack_state!(x, p)
    end

    anderson!(x, F!; maxiter=maxiter, tol=tol)
    unpack_state!(p, x)
    u
end

# ─── GMRES restarted ──────────────────────────────────────────────────────────
# Résout (I-K)x = b0 avec garantie de convergence
# Mémoire : m+5 vecteurs extra (base de Krylov + scratch)
function gmres_solver!(x::AbstractVector, F!, b0::AbstractVector;
                       m::Int=5, maxiter::Int=20, tol=1f-6)
    T  = real(eltype(x))
    TC = eltype(x)

    V      = [similar(x) for _ in 1:m+1]   # base de Krylov (GPU)
    w      = similar(x)                      # scratch matvec
    v_save = similar(x)                      # sauvegarde avant F!
    r      = similar(x)                      # résidu
    x_best = similar(x)
    res_best = T(Inf)

    # (I-K)·v = v - F(v) + b0  (1 appel _propagate! par matvec)
    function Av!(w, v)
        copyto!(v_save, v)
        copyto!(w, v)
        F!(w)                           # w = F(v)
        @. w = v_save - w + b0         # (I-K)v
    end

    copyto!(x_best, x)

    for iter in 1:maxiter                  # restarts
        # r = b0 - (I-K)x = F(x) - x
        copyto!(r, x)
        F!(r)
        @. r = r - x

        β = T(norm(r))
        # @info "iter $iter, res = $β"
        if β < res_best; res_best = β; copyto!(x_best, x); end
        β < tol && break

        @. V[1] = r / β

        H     = zeros(TC, m+1, m)
        j_eff = m

        for j in 1:m                    # Arnoldi (m matvecs par restart)
            Av!(w, V[j])
            for i in 1:j               # Gram-Schmidt modifié
                H[i,j] = dot(V[i], w)  # réduction GPU → scalaire CPU
                @. w -= H[i,j] * V[i]
            end
            h = T(norm(w))
            H[j+1,j] = h
            if h < T(1f-14)            # happy breakdown
                j_eff = j; break
            end
            @. V[j+1] = w / h
        end

        # Moindres carrés (CPU, (j_eff+1)×j_eff, négligeable)
        rhs = zeros(TC, j_eff+1); rhs[1] = β
        y   = H[1:j_eff+1, 1:j_eff] \ rhs

        for j in 1:j_eff               # x ← x + V·y (GPU)
            @. x += y[j] * V[j]
        end
    end

    copyto!(x, x_best)
end

# ─── propagate_gmres! ─────────────────────────────────────────────────────────
function propagate_gmres!(u::HelmholtzField, p::BidirectionalBPM;
                          m::Int=5, maxiter::Int=20, tol=1f-6)
    u_inc_e   = copy(u.electric)
    u_inc_edz = copy(u.electric_dz)

    n_state = length(p.E_tmp) + 2*length(p.ub.electric)
    x  = similar(p.E_tmp, eltype(p.E_tmp), n_state)
    b0 = similar(x)
    pack_state!(x, p)

    function F!(x)
        unpack_state!(p, x)
        copyto!(u.electric,    u_inc_e)
        copyto!(u.electric_dz, u_inc_edz)
        _propagate!(u, p)
        pack_state!(x, p)
    end

    fill!(b0, zero(eltype(b0)))
    F!(b0)

    gmres_solver!(x, F!, b0; m=m, maxiter=maxiter, tol=tol)

    # ← bug était ici : u non mis à jour après la convergence
    unpack_state!(p, x)
    copyto!(u.electric,    u_inc_e)
    copyto!(u.electric_dz, u_inc_edz)
    _propagate!(u, p)      # u ← champ propagé avec le point fixe convergé
    u
end

function propagate!(u::HelmholtzField, p::BidirectionalBPM;
                    method=:gmres, m=10, maxiter=20, tol=1)
    method === :gmres && return propagate_gmres!(u, p; m=m, maxiter=maxiter, tol=tol)
    method === :anderson && return propagate_anderson!(u, p,
        (x, F!; kw...) -> andersonm!(x, F!; m=m, kw...); maxiter=maxiter, tol=tol)
    error("method ∈ {:gmres, :anderson}")
end

# propagate_anderson_m!(u, p, m::Integer; kw...) =
#     propagate_anderson!(u, p, (x, F!; kw2...) -> andersonm!(x, F!; m, kw2...); kw...)

# function propagate!(u::HelmholtzField, p::BidirectionalBPM; maxiter=100, tol=1f-6)
#     propagate_anderson_m!(u, p, 5; maxiter=maxiter, tol=tol)
# end

