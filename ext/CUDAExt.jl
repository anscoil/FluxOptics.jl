module CUDAExt

using AbstractFFTs
using CUDA
using FINUFFT
using FluxOptics.FFTutils
using FluxOptics.OpticalComponents
using FluxOptics.Fields
using ChainRulesCore

function CUDA.cu(u::ScalarField)
    set_field_data(u, cu(u.electric))
end

function Base.unique(x::CuArray)
    unique(Array(x))
end

function FFTutils.make_fft_plans(u::U,
                                 dims::NTuple{N, Integer};
                                 normalize::Bool = true) where {N,
                                                                U <: CuArray{<:Complex}}
    p_ft = plan_fft!(u, dims)
    p_ift = normalize ? plan_ifft!(u, dims) : plan_bfft!(u, dims)
    (; ft = p_ft, ift = p_ift)
end

function Base.exp(A::CuArray{T,2}) where {T}
    cu(exp(collect(A)))
end

function ChainRulesCore.rrule(::typeof(Base.exp), A::CuArray{T,2}) where {T}
    A_cpu = collect(A)
    expA_cpu = exp(A_cpu)
    expA = cu(expA_cpu)
    
    function pullback(∂expA)
        ∂E = collect(unthunk(∂expA))
        n = size(A_cpu, 1)
        Ah = A_cpu'
        M = [Ah ∂E; zeros(T, n, n) Ah]
        ∂A_cpu = exp(M)[1:n, n+1:end]
        return NoTangent(), cu(∂A_cpu)
    end
    return expA, pullback
end

include("cuda/optical_components.jl")

end
