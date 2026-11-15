# ---------------------------------------------------------------------------------------
# Shared tools of the Krylov solvers (GMRES-DR, LGMRES, GCROT, Anderson, IDR(s))
#
# Statistics. IterativeStats holds the number of products by the operator and the
# convergence of the last solve; record! fills it from a convergence history of
# IterativeSolvers.jl.
#
# Windows of stored vectors. Krylov bases, augmentation vectors and histories of
# differences are n × m matrices C used only through Cᴴ·w and target += C·a over their
# first columns. They can be stored in a lower precision than the working one (e.g.
# Complex{Float16} for ComplexF32 work): the products are then done in the working
# precision, as in compressed basis GMRES (Aliaga, Anzt et al., Int. J. High Perform.
# Comput. Appl. 37, 2023), by plain loops on CPU and two KernelAbstractions kernels on
# GPU: Cᴴ·w by lanes of rows and tiles of DOT_TILE columns (w read once per tile), then a
# sum over the lanes, and target += C·a with one work-item per row. BLAS is used when C
# has the working precision. The variant is chosen by dispatch: Array for the host,
# AbstractArray (KernelAbstractions) for any device, so that no GPU package is needed.
# Stored vectors are scaled into the range of the storage type: unit vectors by s0 = √n
# (root mean square 1), others by √n/‖v‖, and by 1 in the working precision.
#
# Small least-squares problems. min ‖g₀ - H̄·y‖ for an upper Hessenberg H̄, reduced to
# triangular form column by column by Givens rotations, as in GMRES.
# ---------------------------------------------------------------------------------------

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

const DOT_LANES = 2048     # partial sums per column of Cᴴ·w on GPU
const DOT_TILE = 8         # columns per work-item of Cᴴ·w (literal 8 in lane_dots!)
const CPU_BLOCK = 4096     # rows per block of the CPU loops, kept in cache

"""Device buffers for the products with a stored window of at most ncols columns."""
struct WindowBuffers{S, M}
    hd::S                 # coefficients on the device, ncols
    partial::M            # partial sums of Cᴴ·w on GPU, lanes × ncols
    dots::M               # their sums, 1 × ncols
end

# Lanes of partial sums of Cᴴ·w: one on the host or in the working precision (BLAS),
# DOT_LANES on a device in a lower precision (KernelAbstractions kernel)
dot_lanes(::Array, storage::Type) = 1
function dot_lanes(x::AbstractArray, storage::Type)
    storage == eltype(x) ? 1 : min(DOT_LANES, length(x))
end

function WindowBuffers(x::AbstractVector, storage::Type, ncols::Integer)
    partial = fill!(similar(x, dot_lanes(x, storage), ncols), zero(eltype(x)))
    WindowBuffers(similar(x, ncols), partial, similar(x, 1, ncols))
end

# Val(true) if the window C is stored in the working precision of the vector w
same_precision(C, w) = Val(eltype(C) == eltype(w))

# hh[1:ncols] = C[:, 1:ncols]ᴴ·w, on the host
function window_dots!(hh, b::WindowBuffers, C, w, ncols)
    window_dots!(hh, b, C, w, ncols, same_precision(C, w))
end

# Working precision: BLAS of the device (OpenBLAS, cuBLAS, ...)
function window_dots!(hh, b::WindowBuffers, C, w, ncols, ::Val{true})
    mul!(view(b.hd, 1:ncols), view(C, :, 1:ncols)', w)
    copyto!(hh, 1, b.hd, 1, ncols)              # device to host: no views
end

# Lower precision on the host: loops by blocks of rows
function window_dots!(hh, ::WindowBuffers, C::Array, w, ncols, ::Val{false})
    cpu_window_dots!(hh, C, w, ncols)
end

# Lower precision on a device: partial sums by lanes, then their sum
function window_dots!(hh, b::WindowBuffers, C::AbstractArray, w, ncols, ::Val{false})
    lanes = size(b.partial, 1)
    window_dots_kernel!(get_backend(C))(b.partial, C, w, ncols;
                                        ndrange = (lanes, cld(ncols, DOT_TILE)))
    lane_sums!(reshape(b.dots, 1, :, 1), reshape(b.partial, lanes, :, 1), ncols)
    copyto!(hh, 1, b.dots, 1, ncols)            # device to host
end

# target += C[:, 1:ncols]·a, with the coefficients a = hh[1:ncols] given on the host
function window_combine!(target, b::WindowBuffers, C, hh, ncols)
    window_combine!(target, b, C, hh, ncols, same_precision(C, target))
end

function window_combine!(target, b::WindowBuffers, C, hh, ncols, ::Val{true})
    copyto!(b.hd, 1, hh, 1, ncols)              # host to device: no views
    mul!(target, view(C, :, 1:ncols), view(b.hd, 1:ncols), true, true)
end

function window_combine!(target, ::WindowBuffers, C::Array, hh, ncols, ::Val{false})
    cpu_window_combine!(target, C, hh, ncols)
end

function window_combine!(target, b::WindowBuffers, C::AbstractArray, hh, ncols,
                         ::Val{false})
    copyto!(b.hd, 1, hh, 1, ncols)              # host to device: no views
    window_combine_kernel!(get_backend(C))(target, C, b.hd, ncols;
                                           ndrange = length(target))
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

# dots[1, c, e] = Σ_l partial[l, c, e] for c ≤ m: one work-item per column and element, in
# place of sum!, which allocates its partial sums at every call when the lanes span
# several blocks (CUDA.jl), i.e. at every iteration of a solver
@kernel function lane_sum_kernel!(dots, @Const(partial))
    c, e = @index(Global, NTuple)
    s = zero(eltype(dots))
    for l in 1:size(partial, 1)
        @inbounds s += partial[l, c, e]
    end
    @inbounds dots[1, c, e] = s
end

function lane_sums!(dots, partial, m)
    lane_sum_kernel!(get_backend(partial))(dots, partial; ndrange = (m, size(partial, 3)))
    dots
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
# Batched windows
#
# A batch of independent problems shares one vector of n entries: blocks of P consecutive
# entries (the spatial dimensions of the fields), block c (from 0) belonging to element
# c mod B + 1, as for arrays of shape (Nx, Ny, batch..., rest...) laid end to end, with P =
# Nx·Ny and B the product of the batch dimensions. A single problem is the case P = n,
# B = 1. The products with a window C then act element by element, each element e with
# its own number of columns ncols[e]:
#   element_dots!:    hh[i, e] = C[:, i]ᴴ·w on the rows of e, i ≤ ncols[e], and
#                     hh[ncols[e] + 1, e] = ‖w‖² on the rows of e (ncols[e] = 0: the norm
#                     only, ncols[e] < 0: element skipped);
#   element_combine!: w += C[:, 1:ncols[e]]·a[:, e] on the rows of e;
#   element_mgs!:     host only, w minus its components on C[:, 1:ncols[e]] by modified
#                     Gram–Schmidt on the rows of e, hh as for element_dots! (the norm
#                     after the projections);
# and gather!, transform!, store! move whole columns element by element.
#
# Host (Array): loops over sub-blocks of at most CPU_BLOCK rows, each inside one element,
# shared among n_tasks Julia tasks (which compose with other tasks, unlike the threads of
# BLAS), with BLAS dot and axpy on the rows of a sub-block in the working precision and
# SIMD loops for a lower storage precision. A single problem in the working precision
# uses BLAS on whole columns instead (its own threads then apply). Device (AbstractArray):
# KernelAbstractions kernels, Cᴴ·w over a grid (lanes × tiles of DOT_TILE columns ×
# elements), the lanes of an element reading its rows with a stride (coalesced inside a
# block), then a sum over the lanes; one work-item per row for the other operations.
# ---------------------------------------------------------------------------------------

"""Batch of B problems in a vector: blocks of P entries, block c to element c mod B + 1."""
struct BatchLayout
    P::Int
    B::Int
end

@inline element_of(r, P, B) = (r - 1) ÷ P % B + 1

"""Buffers of the batched products with a window of at most ncols columns."""
struct BatchBuffers{M, A, D, I, V, H}
    coefs::M              # coefficients on the device, (ncols + 1) × B
    partial::A            # partial sums of Cᴴ·w on a device, lanes × (ncols + 1) × B
    dots::D               # their sums, 1 × (ncols + 1) × B
    ncols::I              # control vectors on the device, B entries each
    cols::I
    modes::I
    scales::V
    host::H               # partial sums of the host tasks, (ncols + 1) × B each
end

function BatchBuffers(x::AbstractVector, layout::BatchLayout, ncols::Integer,
                      n_tasks::Integer)
    T, B = eltype(x), layout.B
    rows = length(x) ÷ B                        # rows of one element
    lanes = x isa Array ? 1 : clamp(DOT_LANES ÷ B, 1, rows)
    host = [zeros(T, ncols + 1, B) for _ in 1:(x isa Array ? n_tasks : 0)]
    BatchBuffers(similar(x, ncols + 1, B),
                 fill!(similar(x, lanes, ncols + 1, B), zero(T)),
                 similar(x, 1, ncols + 1, B), similar(x, Int32, B), similar(x, Int32, B),
                 similar(x, Int32, B), similar(x, real(T), B), host)
end

# Runs f(k) for k = 1:ntasks, as parallel tasks when ntasks > 1
function run_tasks(f, ntasks::Integer)
    ntasks == 1 && return f(1)
    @sync for k in 1:ntasks
        Threads.@spawn f(k)
    end
    nothing
end

# Calls f(rows, e) for the k-th of ntasks shares of the sub-blocks of at most CPU_BLOCK
# rows of a vector of n entries, each sub-block inside the element e
@inline function foreach_subblock(f, n, P, B, k, ntasks)
    nsub = cld(P, CPU_BLOCK)
    total = (n ÷ P) * nsub
    for s in ((k - 1) * total ÷ ntasks):(k * total ÷ ntasks - 1)
        c, j = divrem(s, nsub)
        first = c * P + j * CPU_BLOCK + 1
        f(first:(c * P + min((j + 1) * CPU_BLOCK, P)), c % B + 1)
    end
end

# Host kernels, on the rows of one sub-block: BLAS in the working precision (faster than
# the loops below even on blocks of a few hundred rows, by up to 3× on some CPUs), loops
# otherwise, with real and imaginary parts accumulated separately so that they vectorize
@inline function rows_dot(C::Array{T}, i, w::Array{T}, rows) where {T}
    dot(view(C, rows, i), view(w, rows))
end

@inline rows_norm2(w::Array{<:Complex}, rows) = (v = view(w, rows); real(dot(v, v)))

@inline function rows_axpy!(w::Array{T}, a, C::Array{T}, i, rows) where {T}
    axpy!(a, view(C, rows, i), view(w, rows))
end

@inline function rows_dot(C, i, w, rows)
    T = eltype(w)
    re = im = zero(real(T))
    @inbounds @simd for r in rows
        c, x = T(C[r, i]), w[r]
        re = muladd(real(c), real(x), muladd(imag(c), imag(x), re))
        im = muladd(real(c), imag(x), muladd(-imag(c), real(x), im))
    end
    complex(re, im)
end

@inline function rows_norm2(w, rows)
    s = zero(real(eltype(w)))
    @inbounds @simd for r in rows
        s = muladd(real(w[r]), real(w[r]), muladd(imag(w[r]), imag(w[r]), s))
    end
    s
end

@inline function rows_axpy!(w, a, C, i, rows)
    T = eltype(w)
    ar, ai = real(a), imag(a)
    @inbounds @simd for r in rows
        c = T(C[r, i])
        w[r] = complex(muladd(ar, real(c), muladd(-ai, imag(c), real(w[r]))),
                       muladd(ar, imag(c), muladd(ai, real(c), imag(w[r]))))
    end
end

# Device kernels
@kernel function element_dots_kernel!(partial, C, w, ncols, P, B)
    t, tile, e = @index(Global, NTuple)
    lane_element_dots!(partial, C, w, ncols, P, B, t, tile, e)
end

# Work-item (t, tile, e): rows t, t + lanes, ... of element e, columns DOT_TILE·(tile - 1)
# + 1 to DOT_TILE·tile among 1:ncols[e] and the norm in column ncols[e] + 1
@inline function lane_element_dots!(partial, C, w, ncols, P, B, t, tile, e)
    T = eltype(partial)
    lanes = size(partial, 1)
    ne = Int(ncols[e])
    i0 = (tile - 1) * DOT_TILE
    i0 > ne && return nothing
    nrows = length(w) ÷ B
    Base.Cartesian.@nexprs 8 m -> (acc_m = zero(T))
    l = t - 1
    @inbounds while l < nrows
        q, p = divrem(l, P)
        r = (q * B + e - 1) * P + p + 1
        wr = w[r]
        Base.Cartesian.@nexprs 8 m -> begin
            if i0 + m <= ne
                acc_m += conj(T(C[r, i0 + m])) * wr
            elseif i0 + m == ne + 1
                acc_m += abs2(wr)
            end
        end
        l += lanes
    end
    @inbounds Base.Cartesian.@nexprs 8 m -> (i0 + m <= ne + 1 &&
                                             (partial[t, i0 + m, e] = acc_m))
    nothing
end

@kernel function element_combine_kernel!(target, C, a, ncols, P, B)
    r = @index(Global)
    T = eltype(target)
    e = element_of(r, P, B)
    acc = zero(T)
    @inbounds for i in 1:Int(ncols[e])
        acc += T(C[r, i]) * a[i, e]
    end
    @inbounds target[r] += acc
end

@kernel function element_gather_kernel!(v, C, x, cols, inv_s0, P, B)
    r = @index(Global)
    T = eltype(v)
    c = Int(cols[element_of(r, P, B)])
    @inbounds v[r] = c > 0 ? T(C[r, c]) * inv_s0 : c == 0 ? x[r] : zero(T)
end

@kernel function element_transform_kernel!(w, v, b, modes, P, B)
    r = @index(Global)
    mode = modes[element_of(r, P, B)]
    @inbounds w[r] = mode == 1 ? v[r] - w[r] : mode == 2 ? b[r] - w[r] : zero(eltype(w))
end

@kernel function element_store_kernel!(C, w, cols, scales, P, B)
    r = @index(Global)
    e = element_of(r, P, B)
    c = Int(cols[e])
    @inbounds c > 0 && (C[r, c] = w[r] * scales[e])
end

"""
hh[1:ncols[e], e] = C[:, 1:ncols[e]]ᴴ·w and hh[ncols[e] + 1, e] = ‖w‖², on the rows of
each element e (host matrix hh; ncols[e] = 0: the norm only, ncols[e] < 0: skipped).
"""
function element_dots!(hh, bf::BatchBuffers, C::Array, w, ncols, L::BatchLayout,
                       n_tasks)
    T = eltype(w)
    if L.B == 1 && eltype(C) == T               # single problem: BLAS
        ne = Int(ncols[1])
        ne < 0 && return hh
        ne > 0 && mul!(view(hh, 1:ne, 1), view(C, :, 1:ne)', w)
        hh[ne + 1, 1] = norm(w)^2
        return hh
    end
    run_tasks(n_tasks) do k
        acc = bf.host[k]
        fill!(acc, zero(T))
        foreach_subblock(length(w), L.P, L.B, k, n_tasks) do rows, e
            local m = Int(ncols[e])             # local: not the ne of the BLAS branch
            m < 0 && return
            for i in 1:m
                acc[i, e] += rows_dot(C, i, w, rows)
            end
            acc[m + 1, e] += rows_norm2(w, rows)
        end
    end
    copyto!(hh, first(bf.host))
    for k in 2:n_tasks
        hh .+= bf.host[k]
    end
    hh
end

function element_dots!(hh, bf::BatchBuffers, C::AbstractArray, w, ncols, L::BatchLayout,
                       n_tasks)
    copyto!(bf.ncols, ncols)
    lanes = size(bf.partial, 1)
    tiles = cld(maximum(ncols) + 1, DOT_TILE)
    element_dots_kernel!(get_backend(C))(bf.partial, C, w, bf.ncols, L.P, L.B;
                                         ndrange = (lanes, max(tiles, 1), L.B))
    lane_sums!(bf.dots, bf.partial, max(maximum(ncols) + 1, 1))
    copyto!(hh, 1, bf.dots, 1, length(hh))      # device to host (stale entries unread)
    hh
end

"""target += C[:, 1:ncols[e]]·a[:, e] on the rows of each element e (host matrix a)."""
function element_combine!(target, bf::BatchBuffers, C::Array, a, ncols, L::BatchLayout,
                          n_tasks)
    if L.B == 1 && eltype(C) == eltype(target)  # single problem: BLAS
        ne = Int(ncols[1])
        ne > 0 && mul!(target, view(C, :, 1:ne), view(a, 1:ne, 1), true, true)
        return target
    end
    run_tasks(n_tasks) do k
        foreach_subblock(length(target), L.P, L.B, k, n_tasks) do rows, e
            for i in 1:Int(ncols[e])
                rows_axpy!(target, a[i, e], C, i, rows)
            end
        end
    end
    target
end

function element_combine!(target, bf::BatchBuffers, C::AbstractArray, a, ncols,
                          L::BatchLayout, n_tasks)
    copyto!(bf.coefs, a)
    copyto!(bf.ncols, ncols)
    element_combine_kernel!(get_backend(C))(target, C, bf.coefs, bf.ncols, L.P, L.B;
                                            ndrange = length(target))
    target
end

# Blocks of the rows of element e in a vector of n entries
@inline function element_blocks(n, L::BatchLayout, e)
    ((c * L.P + 1):(c * L.P + L.P) for c in (e - 1):L.B:(n ÷ L.P - 1))
end

"""
Modified Gram–Schmidt on the host, element by element: for i = 1:ncols[e], hh[i, e] =
C[:, i]ᴴ·w on the rows of e, then w -= scale·hh[i, e]·C[:, i] there; finally
hh[ncols[e] + 1, e] = ‖w‖² on the rows of e (ncols[e] = 0: the norm only, ncols[e] < 0:
element skipped). The part of w of an element stays in cache while each column is read
once from memory, against twice per pass of classical Gram–Schmidt. The elements are
shared among the tasks.
"""
function element_mgs!(hh, C::Array, w, ncols, scale, L::BatchLayout, n_tasks)
    T = eltype(w)
    n = length(w)
    ntasks = min(n_tasks, L.B)
    run_tasks(ntasks) do k
        for e in k:ntasks:L.B
            m = Int(ncols[e])
            m < 0 && continue
            for i in 1:m
                d = zero(T)
                for rows in element_blocks(n, L, e)
                    d += rows_dot(C, i, w, rows)
                end
                hh[i, e] = d
                a = -scale * d
                for rows in element_blocks(n, L, e)
                    rows_axpy!(w, a, C, i, rows)
                end
            end
            s = zero(real(T))
            for rows in element_blocks(n, L, e)
                s += rows_norm2(w, rows)
            end
            hh[m + 1, e] = s
        end
    end
    hh
end

"""
Operator inputs v, element by element: column cols[e] of C divided by s0 if positive, x
if zero, 0 if negative.
"""
function gather!(v, bf::BatchBuffers, C::Array, x, cols, s0, L::BatchLayout, n_tasks)
    T = eltype(v)
    inv_s0 = inv(s0)
    run_tasks(n_tasks) do k
        foreach_subblock(length(v), L.P, L.B, k, n_tasks) do rows, e
            c = Int(cols[e])
            if c > 0
                @inbounds @simd for r in rows
                    v[r] = T(C[r, c]) * inv_s0
                end
            elseif c == 0
                @inbounds @simd for r in rows
                    v[r] = x[r]
                end
            else
                @inbounds @simd for r in rows
                    v[r] = zero(T)
                end
            end
        end
    end
    v
end

function gather!(v, bf::BatchBuffers, C::AbstractArray, x, cols, s0, L::BatchLayout,
                 n_tasks)
    copyto!(bf.cols, cols)
    element_gather_kernel!(get_backend(C))(v, C, x, bf.cols, inv(s0), L.P, L.B;
                                   ndrange = length(v))
    v
end

"""
Operator outputs w, element by element: v - w (mode 1), b - w (mode 2), 0 (mode 0).
"""
function transform!(w, bf::BatchBuffers, v::Array, b, modes, L::BatchLayout, n_tasks)
    run_tasks(n_tasks) do k
        foreach_subblock(length(w), L.P, L.B, k, n_tasks) do rows, e
            mode = modes[e]
            if mode == 1
                @inbounds @simd for r in rows
                    w[r] = v[r] - w[r]
                end
            elseif mode == 2
                @inbounds @simd for r in rows
                    w[r] = b[r] - w[r]
                end
            else
                @inbounds @simd for r in rows
                    w[r] = zero(eltype(w))
                end
            end
        end
    end
    w
end

function transform!(w, bf::BatchBuffers, v::AbstractArray, b, modes, L::BatchLayout,
                    n_tasks)
    copyto!(bf.modes, modes)
    element_transform_kernel!(get_backend(w))(w, v, b, bf.modes, L.P, L.B;
                                              ndrange = length(w))
    w
end

"""C[:, cols[e]] = scales[e]·w on the rows of each element e with cols[e] > 0."""
function store!(C::Array, bf::BatchBuffers, w, cols, scales, L::BatchLayout, n_tasks)
    run_tasks(n_tasks) do k
        foreach_subblock(length(w), L.P, L.B, k, n_tasks) do rows, e
            c = Int(cols[e])
            c > 0 || return
            s = scales[e]
            @inbounds @simd for r in rows
                C[r, c] = w[r] * s
            end
        end
    end
    C
end

function store!(C::AbstractArray, bf::BatchBuffers, w, cols, scales, L::BatchLayout,
                n_tasks)
    copyto!(bf.cols, cols)
    copyto!(bf.scales, scales)
    element_store_kernel!(get_backend(C))(C, w, bf.cols, bf.scales, L.P, L.B;
                                  ndrange = length(w))
    C
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
