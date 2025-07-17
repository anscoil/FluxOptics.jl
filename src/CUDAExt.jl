module CUDAExt

using CUDA
using .OpticalComponents

function OpticalComponents.make_fft_plans(
        u::U, dims::NTuple{N, Integer}) where {N, U <: CuArray{<:Complex}}
    p_ft = plan_fft!(u, dims)
    p_ift = plan_ifft!(u, dims)
    (; ft = p_ft, ift = p_ift)
end

end
