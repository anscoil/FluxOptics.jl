module CUDAExt

using CUDA
using ..OpticalComponents

function OpticalComponents.make_fft_plans(
        u::U, dims::NTuple{N, Integer}) where {N, U <: CuArray{<:Complex}}
    p_ft = plan_fft!(u, dims)
    p_ift = plan_ifft!(u, dims)
    (; ft = p_ft, ift = p_ift)
end

function OpticalComponents.compute_phase_gradient!(
        ∂ϕ::P, ∂u::U, u::U
) where {P <: CuArray{<:Real, 2}, U <: CuArray{<:Complex}}
    sdims = Tuple(3:ndims(∂u))
    @views ∂ϕ .= dropdims(sum(imag.(∂u .* conj.(u)), dims = sdims), dims = sdims)
end

end
