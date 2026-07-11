struct ScalarWaveBiProp{M, A, S, K, U, T}  <: AbstractBidirectionalComponent{M}
    trainability::Val{M}
    z::T
    n0::Complex{T}
    n1::Complex{T}
    n2::Complex{T}
    mask::A
    i_01::S
    i_02::S
    i_10::S
    i_20::S
    kernel_n1::BidirectionalKernel{K}
    kernel_n2::BidirectionalKernel{K}
    u_tmp::ScalarWaveField{U}
    conjugate::Bool
end

function ScalarWaveBiProp(u::ScalarWaveField{U}, z::Real, mask::AbstractMatrix,
                          n1::Number, n2::Number;
                          n0::Number = (n1+n2)/2, conjugate::Bool = false
                          ) where {T <: Real, U <: AbstractArray{Complex{T}}}
    ns = size(u)[1:2]
    @assert size(mask) == ns
    z = T(z)
    i_01 = ScalarFlatInterface(u, n0, n1)
    i_02 = ScalarFlatInterface(u, n0, n2)
    i_10 = ScalarFlatInterface(u, n1, n0)
    i_20 = ScalarFlatInterface(u, n2, n0)
    kernel_n1 = BidirectionalKernel(u, z, n1; conjugate)
    kernel_n2 = BidirectionalKernel(u, z, n2; conjugate)
    n0 = Complex{T}(n0)
    n1 = Complex{T}(n1)
    n2 = Complex{T}(n2)
    mask_buf = similar(u.electric, T, size(mask))
    copyto!(mask_buf, mask)
    u_tmp = similar(u)
    ScalarWaveBiProp(Val(Static), z, n0, n1, n2, mask_buf, i_01, i_02, i_10, i_20,
                     kernel_n1, kernel_n2, u_tmp, conjugate)
end

get_n0(p::ScalarWaveBiProp) = p.n0

function alloc_fp_state(u::ScalarWaveField, p::ScalarWaveBiProp)
    (; i_01_state = similar(u.electric),
     i_02_state = similar(u.electric),
     i_10_state = similar(u.electric),
     i_20_state = similar(u.electric),
     E_state = similar(u.electric))
end
