mutable struct IterativeStats
    niter::Int
    solved::Bool
end

IterativeStats() = IterativeStats(0, false)

function record!(stats::IterativeStats, history)
    stats.niter = history.mvps
    stats.solved = history.isconverged
    stats
end

# ---------------------------------------------------------------------------------------
# Shared tools of the Krylov solvers (GMRES-DR, LGMRES, GCROT, Anderson)
#
# Windows of stored vectors. Krylov bases, augmentation vectors and histories of
# differences are n × m matrices C used only through Cᴴ·w and target += C·a over their
# first columns. They can be stored in a lower precision than the working one (e.g.
# Complex{Float16} for ComplexF32 work): the products are then done in the working
# precision, as in compressed basis GMRES (Aliaga, Anzt et al., Int. J. High Perform.
# Comput. Appl. 37, 2023), by plain loops on CPU and two KernelAbstractions kernels on
# GPU: Cᴴ·w by lanes of rows and tiles of DOT_TILE columns (w read once per tile), then a
# sum over the lanes, and target += C·a with one work-item per row. BLAS is used when C
# has the working precision. Stored vectors are scaled into the range of the storage
# type: unit vectors by s0 = √n (root mean square 1), others by √n/‖v‖, and by 1 in the
# working precision.
#
# Small least-squares problems. min ‖g₀ - H̄·y‖ for an upper Hessenberg H̄, reduced to
# triangular form column by column by Givens rotations, as in GMRES.
# ---------------------------------------------------------------------------------------

const DOT_LANES = 2048     # partial sums per column of Cᴴ·w on GPU
const DOT_TILE = 8         # columns per work-item of Cᴴ·w (literal 8 in lane_dots!)
const CPU_BLOCK = 4096     # rows per block of the CPU loops, kept in cache

"""Device buffers for the products with a stored window of at most ncols columns."""
struct WindowBuffers{S, M}
    hd::S                 # coefficients on the device, ncols
    partial::M            # partial sums of Cᴴ·w on GPU, lanes × ncols
    dots::M               # their sums, 1 × ncols
end

function WindowBuffers(x::AbstractVector, storage::Type, ncols::Integer)
    T = eltype(x)
    lanes = storage == T || x isa Array ? 1 : min(DOT_LANES, length(x))
    WindowBuffers(similar(x, ncols), fill!(similar(x, lanes, ncols), zero(T)),
                  similar(x, 1, ncols))
end

# hh[1:ncols] = C[:, 1:ncols]ᴴ·w, on the host
function window_dots!(hh, b::WindowBuffers, C, w, ncols)
    if eltype(C) == eltype(w)
        mul!(view(b.hd, 1:ncols), view(C, :, 1:ncols)', w)
        copyto!(hh, 1, b.hd, 1, ncols)          # device to host: no views
    elseif C isa Array
        cpu_window_dots!(hh, C, w, ncols)
    else
        lanes = size(b.partial, 1)
        window_dots_kernel!(get_backend(C))(b.partial, C, w, ncols;
                                            ndrange = (lanes, cld(ncols, DOT_TILE)))
        sum!(b.dots, b.partial)                 # columns beyond ncols are stale, unread
        copyto!(hh, 1, b.dots, 1, ncols)        # device to host
    end
    hh
end

# target += C[:, 1:ncols]·a, with the coefficients a = hh[1:ncols] given on the host
function window_combine!(target, b::WindowBuffers, C, hh, ncols)
    if C isa Array && eltype(C) != eltype(target)
        return cpu_window_combine!(target, C, hh, ncols)
    end
    copyto!(b.hd, 1, hh, 1, ncols)              # host to device: no views
    if eltype(C) == eltype(target)
        mul!(target, view(C, :, 1:ncols), view(b.hd, 1:ncols), true, true)
    else
        window_combine_kernel!(get_backend(C))(target, C, b.hd, ncols;
                                               ndrange = length(target))
    end
    target
end

# Work-item (t, b) of Cᴴ·w: rows t, t + lanes, ... and columns DOT_TILE·(b - 1) + 1 to
# DOT_TILE·b (at most ncols), accumulated in the working precision; consecutive t read
# consecutive rows (coalesced), and w is read once for the DOT_TILE columns
@inline function lane_dots!(partial, C, w, ncols, t, b)
    T = eltype(partial)
    lanes, n = size(partial, 1), length(w)
    i0 = (b - 1) * DOT_TILE
    Base.Cartesian.@nexprs 8 m -> (acc_m = zero(T))
    r = t
    @inbounds while r <= n
        wr = w[r]
        Base.Cartesian.@nexprs 8 m -> (i0 + m <= ncols &&
                                       (acc_m += conj(T(C[r, i0 + m])) * wr))
        r += lanes
    end
    @inbounds Base.Cartesian.@nexprs 8 m -> (i0 + m <= ncols &&
                                             (partial[t, i0 + m] = acc_m))
    nothing
end

# Row r of target += C[:, 1:ncols]·a, in the working precision
@inline function row_combine!(target, C, a, ncols, r)
    T = eltype(target)
    acc = zero(T)
    @inbounds for i in 1:ncols
        acc += T(C[r, i]) * a[i]
    end
    @inbounds target[r] += acc
    nothing
end

@kernel function window_dots_kernel!(partial, C, w, ncols)
    t, b = @index(Global, NTuple)
    lane_dots!(partial, C, w, ncols, t, b)
end

@kernel function window_combine_kernel!(target, C, a, ncols)
    r = @index(Global)
    row_combine!(target, C, a, ncols, r)
end

# CPU versions, by blocks of rows that stay in cache across the columns
function cpu_window_dots!(g, C, w, ncols)
    T = eltype(w)
    fill!(view(g, 1:ncols), zero(T))
    n = length(w)
    @inbounds for r0 in 1:CPU_BLOCK:n
        r1 = min(r0 + CPU_BLOCK - 1, n)
        for i in 1:ncols
            acc = zero(T)
            @simd for r in r0:r1
                acc += conj(T(C[r, i])) * w[r]
            end
            g[i] += acc
        end
    end
    g
end

function cpu_window_combine!(target, C, a, ncols)
    T = eltype(target)
    n = length(target)
    @inbounds for r0 in 1:CPU_BLOCK:n
        r1 = min(r0 + CPU_BLOCK - 1, n)
        for i in 1:ncols
            ai = T(a[i])
            @simd for r in r0:r1
                target[r] += T(C[r, i]) * ai
            end
        end
    end
    target
end

# ---------------------------------------------------------------------------------------
# Stored vectors
# ---------------------------------------------------------------------------------------

"""Scale of unit vectors stored with element type Tc for n entries in precision T."""
unit_scale(::Type{Tc}, ::Type{T}, n) where {Tc, T} = Tc == T ? one(real(T)) :
                                                     sqrt(real(T)(n))

"""Scale of v stored in C: 1 in the working precision, √n/‖v‖ otherwise."""
function storage_scale(C, v)
    R = real(eltype(v))
    eltype(C) == eltype(v) && return one(R)
    nv = norm(v)
    iszero(nv) ? one(R) : sqrt(R(length(v))) / nv
end

"""Column j of C in the working precision, into v: C[:, j] / s."""
load_stored!(v, C, j, s) = (v .= view(C, :, j) ./ s)

"""v stored as column j of C, multiplied by s (and rounded to the storage type)."""
store_scaled!(C, j, v, s) = (view(C, :, j) .= v .* s; C)

"""
w minus its components on the unit vectors stored in C[:, 1:ncols] (as s0·v), by two
passes of classical Gram–Schmidt; the components are added to h (host, length ncols).
hh is a host buffer of length at least ncols.
"""
function orthogonalize_window!(w, h, hh, buffers::WindowBuffers, C, s0, ncols)
    ncols == 0 && return w
    hc = view(hh, 1:ncols)
    for pass in 1:2
        window_dots!(hh, buffers, C, w, ncols)  # s0·Cᴴ·w
        hc ./= s0
        h .+= hc
        hc .*= -inv(s0)                         # w -= C·(Cᴴ·w / s0)
        window_combine!(w, buffers, C, hh, ncols)
    end
    w
end

# ---------------------------------------------------------------------------------------
# Hessenberg least squares
# ---------------------------------------------------------------------------------------

"""
Column j of the upper Hessenberg H brought to triangular form in Rq by the rotations of
the previous columns, then a new rotation that zeroes H[j+1, j], also applied to g.
Returns |g[j+1]|, the residual norm of min ‖g₀ - H̄[1:j+1, 1:j]·y‖.
"""
function hessenberg_column!(Rq, g, rot, H, j)
    col = view(Rq, 1:(j + 1), j)
    copyto!(col, view(H, 1:(j + 1), j))
    for i in 1:(j - 1)
        lmul!(rot[i], col)
    end
    G, r = givens(col[j], col[j + 1], j, j + 1)
    col[j], col[j + 1] = r, zero(r)
    rot[j] = G
    lmul!(G, view(g, 1:(j + 1)))
    abs(g[j + 1])
end

"""Solution y[1:j] = Rq[1:j, 1:j]⁻¹·g[1:j] of the reduced problem, returned as a view."""
function hessenberg_solve!(y, Rq, g, j)
    yj = view(y, 1:j)
    copyto!(yj, view(g, 1:j))
    ldiv!(UpperTriangular(view(Rq, 1:j, 1:j)), yj)
end

include("idrs_solver.jl")
include("anderson_solver.jl")
include("gmres-dr_solver.jl")
include("gcrot_solver.jl")
