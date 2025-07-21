using AbstractFFTs
using FFTW
using EllipsisNotation

function make_fft_plans(
        u::U, dims::NTuple{N, Integer}) where {N, U <: AbstractArray{<:Complex}}
    p_ft = plan_fft!(u, dims, flags = FFTW.MEASURE)
    p_ift = plan_ifft!(u, dims, flags = FFTW.MEASURE)
    (; ft = p_ft, ift = p_ift)
end

struct ASProp{M, T, F, K, V, P} <: AbstractPropagator{M}
    f_vec::V
    p_ker::K
    p_f::P
    filter::F

    function ASProp(
            U::Type{<:AbstractArray{Complex{T}, N}},
            dims::NTuple{N, Integer},
            dx::Real,
            dy::Real,
            λ::Real,
            z::Real;
            filter::Union{Nothing, AbstractArray{<:Real, 2}} = nothing
    ) where {N, T <: Real}
        @assert N >= 2
        K = adapt_dim(U, 2)
        nx, ny = dims
        fxv = fftfreq(nx, T(1/dx))
        fyv = fftfreq(ny, T(1/dy))
        f_vec = (; x = fxv, y = fyv)
        p_ker = (@. exp(im*(2*π*z)*sqrt((1/λ^2 + 0im) - fxv^2 - (fyv')^2))) |> K
        A_plan = U(undef, dims)
        p_f = make_fft_plans(A_plan, (1, 2))
        filter = isnothing(filter) ? nothing : filter |> K
        F = typeof(filter)
        V = typeof(f_vec)
        P = typeof(p_f)
        new{Static, T, F, K, V, P}(f_vec, p_ker, p_f, filter)
    end

    function ASProp(u::U,
            dx::Real,
            dy::Real,
            λ::Real,
            z::Real;
            filter::Union{Nothing, AbstractArray{<:Real, 2}} = nothing
    ) where {U <: AbstractArray{<:Complex}}
        ASProp(typeof(u), size(u), dx, dy, λ, z; filter = filter)
    end
end

function kernel(prop::AbstractPropagator)
    prop.p_ker
end

function conjugate_kernel(prop::AbstractPropagator)
    conj.(prop.p_ker)
end

function apply_kernel!(u, prop::AbstractPropagator, ::Type{Forward})
    u .*= kernel(prop)
end

function apply_kernel!(u, prop::AbstractPropagator, ::Type{Backward})
    u .*= conjugate_kernel(prop)
end

function kernel_expr(as_prop::ASProp{M, T}, λ, z) where {M, T}
    fx, fy = as_prop.f_vec.x, as_prop.f_vec.y'
    k² = complex(inv(T(λ)^2))
    im * T(2π*z) .* sqrt.(k² .- fx .* fx .- fy .* fy)
end

function apply_kernel!(u, as_prop::ASProp, λ, z, ::Type{Forward})
    u .*= exp.(kernel_expr(as_prop, λ, z))
end

function apply_kernel!(u, as_prop::ASProp, λ, z, ::Type{Backward})
    u .*= conj.(exp.(kernel_expr(as_prop, λ, z)))
end

function mul_filter!(u, as_prop::ASProp{M, T, Nothing}) where {M, T}
    u
end

function mul_filter!(u, as_prop::ASProp{M, T, F}) where {M, T, F <: AbstractArray{T}}
    u .*= as_prop.filter
end

function _propagate_core!(apply_kernel_fn, u, as_prop::ASProp)
    as_prop.p_f.ft * u
    apply_kernel_fn()
    mul_filter!(u, as_prop)
    as_prop.p_f.ift * u
end

function propagate!(u, as_prop::ASProp, direction::Type{<:Direction})
    _propagate_core!(u, as_prop) do
        apply_kernel!(u, as_prop, direction)
    end
end

function propagate!(u, as_prop::ASProp, λ, z, direction::Type{<:Direction})
    _propagate_core!(u, as_prop) do
        apply_kernel!(u, as_prop, λ, z, direction)
    end
end

struct RSProp{M, T, K, U, P} <: AbstractPropagator{M}
    p_ker::K
    u_tmp::U
    p_f::P

    function RSProp(U::Type{<:AbstractArray{Complex{T}, N}},
            dims::NTuple{N, Integer}, dx::Real, dy::Real, λ::Real, z::Real
    ) where {N, T <: Real}
        @assert N >= 2
        K = adapt_dim(U, 2)
        nx, ny = dims
        Nx, Ny = 2*nx-1, 2*ny-1
        x_vec = circshift((1 - nx):(nx - 1), nx) .* dx
        y_vec = circshift((1 - ny):(ny - 1), ny) .* dy
        k = 2*π/λ
        r_vec = @. sqrt(x_vec^2 + (y_vec')^2 + z^2)
        p_ker = (@. (dx*dy/2π)*(exp(im*k*r_vec)/r_vec)*(z/r_vec)*(1/r_vec-im*k)) |> K
        fft!(p_ker)
        A_plan = U(undef, (Nx, Ny, dims[3:end]...))
        p_f = make_fft_plans(A_plan, (1, 2))
        P = typeof(p_f)
        new{Static, T, K, U, P}(p_ker, A_plan, p_f)
    end

    function RSProp(u::U,
            dx::Real, dy::Real, λ::Real, z::Real
    ) where {U <: AbstractArray{<:Complex}}
        RSProp(U, size(u), dx, dy, λ, z)
    end
end

function propagate!(u, rs_prop::RSProp, direction::Type{<:Direction})
    nx, ny = size(u)
    rs_prop.u_tmp .= 0
    u_view = @view rs_prop.u_tmp[1:nx, 1:ny, ..]
    u_view .= u
    rs_prop.p_f.ft * rs_prop.u_tmp
    apply_kernel!(rs_prop.u_tmp, rs_prop, direction)
    rs_prop.p_f.ift * rs_prop.u_tmp
    @views u .= u_view
    u
end

function backpropagate!(u, p::AbstractPropagator, direction::Type{<:Direction})
    propagate!(u, p, reverse(direction))
end
