struct ScalarFresnelKernel{K}
    a1::K
    a2::K
    r21::K
    t12::K
    t21::K
end

Adapt.@adapt_structure ScalarFresnelKernel

struct ScalarFlatInterface{K, T}  <: AbstractBidirectionalComponent
    n1::Complex{T}
    n2::Complex{T}
    kernel::ScalarFresnelKernel{K}
end

function ScalarFlatInterface(u::ScalarWaveField{U}, n1::Number, n2::Number,
                             ) where {T <: Real, U <: AbstractArray{Complex{T}}}
    n1 = Complex{T}(n1)
    n2 = Complex{T}(n2)
    a1 = im * compute_kz(u, n1)
    a2 = im * compute_kz(u, n2)
    r21 = compute_fresnel_r12(u, n2, n1)
    t12 = compute_fresnel_t12(u, n1, n2)
    t21 = compute_fresnel_t12(u, n2, n1)
    kernel = ScalarFresnelKernel(a1, a2, r12, t12, t21)
    ScalarFlatInterface(n1, n2, kernel)
end

get_n0_left(p::ScalarFlatInterface) = p.n1

get_n0_right(p::ScalarFlatInterface) = p.n2

function initial_state(u::ScalarWaveField, p::ScalarFlatInterface)
    (; E_state = zero(u.electric))
end

@kernel function scalar_flat_interface_kernel!(electric, electric_dz, E_state,
                                               kernel, ::Val{forward}) where {forward}
    s = forward ? 1 : -1
    kernel_a1 = forward ? kernel.a1 : kernel.a2
    kernel_a2 = forward ? kernel.a2 : kernel.a1
    kernel_t12 = forward ? kernel.t12 : kernel.t21
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(electric)[3:end])
        a1 = _get_val(kernel_a1, I, J)
        a2 = _get_val(kernel_a2, I, J)
        r21 = _get_val(kernel.r21, I, J)
        t12 = _get_val(kernel_t12, I, J)
        E_val = electric[I,J]
        dE_val = electric_dz[I,J]
        E2 = E_state[I,J]
        E1 = 0.5 * (E_val + s * dE_val / a1)
        E_state[I,J] = E1
        E1 = t12 * E1 + s * r21 * E2
        electric[I,J] = E1 + E2
        electric_dz[I,J] =  a2 * s * (E1 - E2)
    end
end

function propagate!(u::ScalarWaveField, state, p::ScalarFlatInterface)
    backend = get_backend(u.electric)
    scalar_flat_interface_kernel!(backend)(
        u.electric, u.electric_dz, state.E_state, p.kernel, Val(true);
        ndrange = size(u.electric)[1:2])
    u
end 

function inverse_propagate!(u::ScalarWaveField, state, p::ScalarFlatInterface)
    backend = get_backend(u.electric)
    scalar_flat_interface_kernel!(backend)(
        u.electric, u.electric_dz, state.E_state, p.kernel, Val(false);
        ndrange = size(u.electric)[1:2])
    u
end
