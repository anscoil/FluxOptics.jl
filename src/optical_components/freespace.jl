using AbstractFFTs
using FFTW
using EllipsisNotation

function make_fft_plans(
        u::U, dims::NTuple{N, Integer}) where {N, U <: AbstractArray{<:Complex}}
    p_ft = plan_fft!(u, dims, flags = FFTW.MEASURE)
    p_ift = plan_ifft!(u, dims, flags = FFTW.MEASURE)
    (; ft = p_ft, ift = p_ift)
end

function prepare_fft_plans(U, dims)
    A_plan = U(undef, dims)
    p_f = make_fft_plans(A_plan, (1, 2))
    typeof(p_f), p_f
end

function prepare_fvec(T, nx, ny, dx, dy)
    fxv = fftfreq(nx, T(1/dx))
    fyv = fftfreq(ny, T(1/dy))
    f_vec = (; x = fxv, y = fyv)
    typeof(f_vec), f_vec
end

function prepare_filter(U, filter)
    if isnothing(filter)
        Nothing, nothing
    else
        adapt_dim(U, 2, real), filter |> F
    end
end

function as_kernel(fx, fy, λ, z)
    f² = complex(1/λ^2)
    exp(im*2π*z*sqrt(f² - fx*fx - (fy*fy)))
end

struct ASProp{M, K, F, T, V, P} <: AbstractPropagator{M}
    f_vec::V
    p_ker::K
    z::T
    p_f::P
    filter::F

    function ASProp(
            U::Type{<:AbstractArray{Complex{T}, N}},
            dims::NTuple{N, Integer},
            dx::Real,
            dy::Real,
            lambdas::Union{Real, Tuple},
            z::Real;
            filter::Union{Nothing, AbstractArray{<:Real, 2}} = nothing
    ) where {N, T <: Real}
        @assert N >= 2
        nx, ny = dims
        V, f_vec = prepare_fvec(T, nx, ny, dx, dy)
        P, p_f = prepare_fft_plans(U, dims)
        F, filter = prepare_filter(U, filter)
        if isa(lambdas, Real)
            λ = lambdas
            K = adapt_dim(U, 2)
            p_ker = (@. as_kernel(f_vec.x, f_vec.y', λ, z)) |> K
        elseif lambdas == ()
            K = Nothing
            p_ker = nothing
        else
            Ke = adapt_dim(U, 2)
            K = Dict{T, Ke}
            p_ker = K()
            for λ in lambdas
                p_ker_e = (@. as_kernel(f_vec.x, f_vec.y', λ, z)) |> Ke
                p_ker[T(λ)] = p_ker_e
            end
        end
        new{Static, K, F, T, V, P}(f_vec, p_ker, T(z), p_f, filter)
    end

    function ASProp(
            U::Type{<:AbstractArray{Complex{T}, N}},
            dims::NTuple{N, Integer},
            dx::Real,
            dy::Real,
            z::Real;
            filter::Union{Nothing, AbstractArray{<:Real, 2}} = nothing
    ) where {N, T <: Real}
        ASProp(U, dims, dx, dy, (), z; filter = filter)
    end

    function ASProp(u::U,
            dx::Real,
            dy::Real,
            lambdas::Union{Real, Tuple},
            z::Real;
            filter::Union{Nothing, AbstractArray{<:Real, 2}} = nothing
    ) where {U <: AbstractArray{<:Complex}}
        ASProp(typeof(u), size(u), dx, dy, lambdas, z; filter = filter)
    end

    function ASProp(u::U,
            dx::Real,
            dy::Real,
            z::Real;
            filter::Union{Nothing, AbstractArray{<:Real, 2}} = nothing
    ) where {U <: AbstractArray{<:Complex}}
        ASProp(typeof(u), size(u), dx, dy, z; filter = filter)
    end

    function ASProp(u::ScalarField,
            dx::Real,
            dy::Real,
            lambdas::Real,
            z::Real;
            filter::Union{Nothing, AbstractArray{<:Real, 2}} = nothing)
        @assert all(==(lambdas), u.lambdas)
        ASProp(u.data, dx, dy, lambdas, z; filter = filter)
    end

    function ASProp(u::ScalarField,
            dx::Real,
            dy::Real,
            z::Real;
            filter::Union{Nothing, AbstractArray{<:Real, 2}} = nothing,
            kernel_cache = false
    )
        if kernel_cache
            ASProp(u.data, dx, dy, Tuple(Set(u.lambdas)), z; filter = filter)
        else
            ASProp(u.data, dx, dy, (), z; filter = filter)
        end
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

function apply_kernel!(u, as_prop::ASProp{M, K}, λ, z,
        ::Type{Forward}) where {M, K <: Union{Nothing, <:AbstractArray}}
    fx, fy = as_prop.f_vec.x, as_prop.f_vec.y'
    @. u *= as_kernel(fx, fy, λ, z)
end

function apply_kernel!(u, as_prop::ASProp{M, K}, λ, z,
        ::Type{Backward}) where {M, K <: Union{Nothing, <:AbstractArray}}
    fx, fy = as_prop.f_vec.x, as_prop.f_vec.y'
    @. u *= conj(as_kernel(fx, fy, λ, z))
end

function apply_kernel!(u, as_prop::ASProp, λ, direction::Type{<:Direction})
    apply_kernel!(u, as_prop, λ, as_prop.z, direction)
end

function apply_kernel!(
        u::ScalarField, as_prop::ASProp{M, <:Dict}, ::Type{Forward}) where {M}
    inds = CartesianIndices(size(u.data)[3:end])
    for i in inds
        @views u.data[:,:,i] .*= as_prop.p_ker[u.lambdas[i]]
    end
    u
end

function apply_kernel!(
        u::ScalarField, as_prop::ASProp{M, <:Dict}, ::Type{Backward}) where {M}
    inds = CartesianIndices(size(u.data)[3:end])
    for i in inds
        @views u.data[:,:,i] .*= conj.(as_prop.p_ker[u.lambdas[i]])
    end
    u
end

function mul_filter!(u, as_prop::ASProp{M, K, Nothing}) where {M, K}
    u
end

function mul_filter!(u, as_prop::ASProp{M, K, F}) where {M, K, F <: AbstractArray}
    u .*= as_prop.filter
end

function _propagate_core!(apply_kernel_fn, u, as_prop::ASProp)
    as_prop.p_f.ft * u
    apply_kernel_fn()
    mul_filter!(u, as_prop)
    as_prop.p_f.ift * u
end

function propagate!(u::AbstractArray, as_prop::ASProp{M, <:AbstractArray},
        direction::Type{<:Direction}) where {M}
    _propagate_core!(u, as_prop) do
        apply_kernel!(u, as_prop, direction)
    end
end

function propagate!(u::AbstractArray, as_prop::ASProp, λ, z, direction::Type{<:Direction})
    _propagate_core!(u, as_prop) do
        apply_kernel!(u, as_prop, λ, z, direction)
    end
end

function propagate!(u::AbstractArray, as_prop::ASProp, λ, direction::Type{<:Direction})
    propagate!(u, as_prop, λ, as_prop.z, direction)
end

function propagate!(u::ScalarField, as_prop::ASProp{M, <:AbstractArray},
        direction::Type{<:Direction}) where {M}
    propagate!(u.data, as_prop, direction)
    u
end

function propagate!(u::ScalarField, as_prop::ASProp{M, <:Dict},
        direction::Type{<:Direction}) where {M}
    _propagate_core!(u.data, as_prop) do
        apply_kernel!(u, as_prop, direction)
    end
    u
end

function propagate!(u::ScalarField, as_prop::ASProp{M, Nothing},
        direction::Type{<:Direction}) where {M}
    propagate!(u.data, as_prop, u.lambdas, as_prop.z, direction)
    u
end

function propagate!(u::ScalarField, as_prop::ASProp, z, direction::Type{<:Direction})
    propagate!(u.data, as_prop, u.lambdas, z, direction)
    u
end

struct RSProp{M, K, T, U, P} <: AbstractPropagator{M}
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
        new{Static, K, T, U, P}(p_ker, A_plan, p_f)
    end

    function RSProp(u::U,
            dx::Real, dy::Real, λ::Real, z::Real
    ) where {U <: AbstractArray{<:Complex}}
        RSProp(U, size(u), dx, dy, λ, z)
    end
end

function propagate!(u::AbstractArray, rs_prop::RSProp, direction::Type{<:Direction})
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
