module FFTutils

using AbstractFFTs
using ..Fields: ScalarField
import FFTW

export compute_ft!, compute_ift!, FFTPlans
export make_fft_plan, make_fft_plans
export plan_czt, plan_czt!

const FFTPlans = NamedTuple{(:ft, :ift), <:Tuple{AbstractFFTs.Plan, AbstractFFTs.Plan}}

function make_fft_plans(
        u::U, dims::NTuple{N, Integer}) where {N, U <: AbstractArray{<:Complex}}
    p_ft = plan_fft!(u, dims, flags = FFTW.MEASURE)
    p_ift = plan_ifft!(u, dims, flags = FFTW.MEASURE)
    (; ft = p_ft, ift = p_ift)
end

function compute_ft!(p_f::FFTPlans, u::ScalarField)
    p_f.ft * u.electric
    u
end

function compute_ift!(p_f::FFTPlans, u::ScalarField)
    p_f.ift * u.electric
    u
end

function prepare_czt_data(x::AbstractArray, dim::Integer, a, w, center_on_grid = false)
    n = size(x, dim)
    N = 2*n-1
    U = similar(typeof(x), 1)
    k_range = (0:(N - 1))
    if center_on_grid
        k_range = k_range .- (n-1)/2
    end
    f_range = N * fftfreq(N, 1)
    ρa, θa = a[dim]
    ρw, θw = w[dim]
    input_chirp = [exp(k^2/2*log(ρw) - k*log(ρa))*exp(im*(k^2/2*θw - k*θa))
                   for k in k_range] |> U
    output_chirp = [exp(k^2/2*log(ρw))*exp(im*(k^2/2*θw)) for k in k_range] |> U
    h = [exp(-k^2/2*log(ρw))*exp(-im*(k^2/2*θw)) for k in f_range] |> U
    shape = ntuple(k -> k == dim ? N : 1, ndims(x))
    (reshape(input_chirp, shape), reshape(output_chirp, shape), reshape(h, shape))
end

struct CZTPlan{U, B, N, P}
    p_f::P
    input_chirp::U
    output_chirp::U
    h_f::U
    x_tmp::U

    function CZTPlan(x::U,
            dims::NTuple{Nd, Integer};
            a::NTuple{N} = ntuple(_ -> (1, 0), N),
            w::NTuple{N} = ntuple(k -> (1, -2π/size(x, k)), N),
            inplace = false,
            center_on_grid::Bool = false
    ) where {Nd, N, T <: Real, U <: AbstractArray{Complex{T}, N}}
        @assert unique(dims) == collect(dims)
        @assert all(k -> k >= 1 && k <= N, dims)
        input_chirp = T(1)
        output_chirp = T(1)
        h = T(1)
        for dim in dims
            in_1d, out_1d, h_1d = prepare_czt_data(x, dim, a, w, center_on_grid)
            input_chirp = input_chirp .* in_1d
            output_chirp = output_chirp .* out_1d
            h = h .* h_1d
        end
        dims_tmp = ntuple(k -> k in dims ? 2*size(x, k) - 1 : size(x, k), N)
        x_tmp = similar(x, dims_tmp)
        p_f = make_fft_plans(x_tmp, dims)
        new{U, Val{inplace}, Nd, typeof(p_f)}(
            p_f, input_chirp, output_chirp, fft(h, dims), x_tmp)
    end
end

struct CZTAdjointPlan{C}
    p::C
    function CZTAdjointPlan(p::C) where {C <: CZTPlan}
        new{C}(p)
    end
end

Base.adjoint(p::CZTPlan) = CZTAdjointPlan(p)

function convert_czt_arg(a, N)
    if isa(a, Number)
        return ntuple(_ -> (abs(a), angle(a)), N)
    elseif isa(a, NTuple{N, Number})
        return ntuple(i -> (abs(a[i]), angle(a[i])), N)
    elseif isa(a, NTuple{N, Tuple{Real, Real}})
        return a
    else
        error("Wrong argument")
    end
end

function plan_czt(x, dims, a, w; kwargs...)
    N = ndims(x)
    a = convert_czt_arg(a, N)
    w = convert_czt_arg(w, N)
    CZTPlan(x, dims; a = a, w = w, kwargs...)
end

function plan_czt(x, dims; kwargs...)
    CZTPlan(x, dims; kwargs...)
end

function plan_czt!(x, dims, args...; center_on_grid = false)
    plan_czt(x, dims, args...; inplace = true, center_on_grid = center_on_grid)
end

function czt!(p::CZTPlan{U}, x::U) where {U}
    p.x_tmp .= 0
    x_view = view(p.x_tmp, axes(x)...)
    copyto!(x_view, x)
    p.x_tmp .*= p.input_chirp
    p.p_f.ft * p.x_tmp
    p.x_tmp .*= p.h_f
    p.p_f.ift * p.x_tmp
    p.x_tmp .*= p.output_chirp
    copyto!(x, x_view)
    x
end

function iczt!(p::CZTPlan{U}, x::U) where {U}
    error("Not implemented")
end

function czt!(pa::CZTAdjointPlan{C}, x::U) where {U, C <: CZTPlan{U}}
    p = pa.p
    p.x_tmp .= 0
    x_view = view(p.x_tmp, axes(x)...)
    copyto!(x_view, x)
    p.x_tmp .*= conj.(p.output_chirp)
    p.p_f.ft * p.x_tmp
    p.x_tmp .*= conj.(p.h_f)
    p.p_f.ift * p.x_tmp
    p.x_tmp .*= conj.(p.input_chirp)
    copyto!(x, x_view)
    x
end

function Base.:*(p::CZTPlan{U, Val{true}}, x::U) where {U}
    czt!(p, x)
end

function Base.:*(p::CZTPlan{U, Val{false}}, x::U) where {U}
    xr = copy(x)
    czt!(p, xr)
end

function Base.:*(pa::CZTAdjointPlan{C}, x::U) where {U, C <: CZTPlan{U, Val{true}}}
    czt!(pa, x)
end

function Base.:*(pa::CZTAdjointPlan{C}, x::U) where {U, C <: CZTPlan{U, Val{false}}}
    xr = copy(x)
    czt!(pa, xr)
end

function Base.:\(p::CZTPlan{U, Val{true}}, x::U) where {U}
    iczt!(p, x)
end

function Base.:\(p::CZTPlan{U, Val{false}}, x::U) where {U}
    xr = copy(x)
    iczt!(p, xr)
end

end
