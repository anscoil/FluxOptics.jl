using AbstractFFTs
using FFTW
using EllipsisNotation

function apply_kernel!(prop::AbstractOpticalComponent{U}, u::U, ::Type{Forward}) where {U}
    u .*= kernel(prop)
end

function apply_kernel!(prop::AbstractOpticalComponent{U}, u::U, ::Type{Backward}) where {U}
    u .*= conjugate_kernel(prop)
end

struct ASProp{T} <: AbstractOpticalComponent{T}
    fx_vec
    fy_vec
    p_ker
    p_ft
    p_ift
    filter

    function ASProp(
            dims::NTuple{N, Integer},
            dx::Real,
            dy::Real,
            λ::Real,
            z::Real;
            filter::Union{Nothing, AbstractArray{<:Real, 2}} = nothing,
            U::Type{<:AbstractArray{<:Complex}} = Array{ComplexF64, N}
    ) where {N}
        @assert N >= 2
        K = adapt_2D(U)
        nx, ny = dims
        fxv = fftfreq(nx, 1/dx)
        fyv = fftfreq(ny, 1/dy)
        p_ker = K(exp.(im*(2*π*z) .* sqrt.((1/λ^2 + 0im) .- fxv .^ 2 .- (fyv') .^ 2)))
        A_plan = U(undef, dims)
        if isa(A_plan, Array)
            p_ft = plan_fft!(A_plan, (1, 2), flags = FFTW.MEASURE)
            p_ift = plan_ifft!(A_plan, (1, 2), flags = FFTW.MEASURE)
        else
            p_ft = plan_fft!(A_plan, (1, 2))
            p_ift = plan_ifft!(A_plan, (1, 2))
        end
        new{Static}(fxv, fyv, p_ker, p_ft, p_ift, filter)
    end

    function ASProp(u::U,
            dx::Real,
            dy::Real,
            λ::Real,
            z::Real;
            filter::Union{Nothing, AbstractArray{<:Real, 2}} = nothing
    ) where {U <: AbstractArray{<:Complex}}
        ASProp(size(u), dx, dy, λ, z; filter = filter, U = typeof(u))
    end
end

function kernel(as_prop::ASProp)
    as_prop.p_ker
end

function conjugate_kernel(as_prop::ASProp)
    conj.(as_prop.p_ker)
end

function apply_kernel!(as_prop::ASProp{Static}, u, λ::Real, z::Real, ::Type{Forward})
    u .*= exp.(im .* (2*π*z) .*
               sqrt.((1/λ^2 + 0im) .- as_prop.fx_vec .^ 2 .- (as_prop.fy_vec') .^ 2))
end

function apply_kernel!(as_prop::ASProp{Static}, u, λ::Real, z::Real, ::Type{Backward})
    u .*= conj.(exp.(im .* (2*π*z) .*
                     sqrt.((1/λ^2 + 0im) .- as_prop.fx_vec .^ 2 .- (as_prop.fy_vec') .^ 2)))
end

function propagate!(u, as_prop::ASProp{Static}; direction::Type{<:Direction} = Forward)
    as_prop.p_ft * u
    apply_kernel!(as_prop, u, direction)
    if !isnothing(as_prop.filter)
        u .*= filter
    end
    as_prop.p_ift * u
end

function propagate!(u, as_prop::ASProp{Static}, λ::Real, z::Real;
        direction::Type{<:Direction} = Forward)
    as_prop.p_ft * u
    apply_kernel!(as_prop, u, λ, z, direction)
    if !isnothing(as_prop.filter)
        u .*= filter
    end
    as_prop.p_ift * u
end

struct RSProp{T} <: AbstractOpticalComponent{T}
    p_ker
    p_ker_c
    u_tmp
    p_ft
    p_ift

    function RSProp(dims::NTuple{N, Integer}, dx::Real, dy::Real, λ::Real, z::Real;
            U::Type{<:AbstractArray{<:Complex, N}} = Array{ComplexF64, N}
    ) where {N}
        @assert N >= 2
        K = adapt_2D(U)
        nx, ny = dims
        Nx, Ny = 2*nx-1, 2*ny-1
        X_vec = circshift((1 - nx):(nx - 1), nx) .* dx
        Y_vec = circshift((1 - ny):(ny - 1), ny) .* dy
        k = 2*π/λ
        R_vec = sqrt.(X_vec .^ 2 .+ (Y_vec') .^ 2 .+ z^2)
        p_ker = K((dx*dy/2π)*(exp.(im*k .* R_vec) ./ R_vec) .* (z ./ R_vec) .*
                  (1.0 ./ R_vec .- im*k))
        A_plan = U(undef, (Nx, Ny, dims[3:end]...))
        if isa(A_plan, Array)
            p_ft = plan_fft!(A_plan, (1, 2), flags = FFTW.MEASURE)
            p_ift = plan_ifft!(A_plan, (1, 2), flags = FFTW.MEASURE)
        else
            p_ft = plan_fft!(A_plan, (1, 2))
            p_ift = plan_ifft!(A_plan, (1, 2))
        end
        new{Static}(fft(p_ker), fft(conj(p_ker)), A_plan, p_ft, p_ift)
    end

    function RSProp(
            u::U, dx::Real, dy::Real, λ::Real, z::Real
    ) where {U <: AbstractArray{<:Complex}}
        RSProp(size(u), dx, dy, λ, z; U = typeof(u))
    end
end

function kernel(rs_prop::RSProp)
    rs_prop.p_ker
end

function conjugate_kernel(rs_prop::RSProp)
    rs_prop.p_ker_c
end

function propagate!(u, rs_prop::RSProp{Static}; direction::Type{<:Direction} = Forward)
    nx, ny = size(u)
    rs_prop.u_tmp .= 0
    rs_prop.u_tmp[1:nx, 1:ny, ..] .= u
    rs_prop.p_ft * rs_prop.u_tmp
    apply_kernel!(rs_prop, rs_prop.u_tmp, direction)
    rs_prop.p_ift * rs_prop.u_tmp
    @views u .= rs_prop.u_tmp[1:nx, 1:ny, ..]
    u
end
