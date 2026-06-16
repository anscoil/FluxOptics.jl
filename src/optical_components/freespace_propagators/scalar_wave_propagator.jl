struct BidirectionalKernel{K}
    a::K
    exp_a_p::K
    exp_a_m::K
end

Adapt.@adapt_structure BidirectionalKernel

struct ScalarWavePropagator{K, T}  <: AbstractBidirectionalComponent
    z::T
    n0::Complex{T}
    kernel::BidirectionalKernel{K}
end


function ScalarWavePropagator(u::ScalarWaveField{U}, z::Real, n0::Number
                              ) where {T <: Real, U <: AbstractArray{Complex{T}}}
    ns = size(u)[1:2]
    z = T(z)
    n0 = Complex{T}(n0)
    a = im * compute_kz(u, n0)
    exp_a_p = @. exp(a * z)
    exp_a_m = @. exp(-a * z)
    kernel = BidirectionalKernel(a, exp_a_p, exp_a_m)
    ScalarWavePropagator(z, exp_a_p, exp_a_m)
end

get_n0(p::ScalarWavePropagator) = p.n0

function initial_state(u::ScalarWaveField, p::ScalarWavePropagator)
    (; E_state = similar(u.electric))
end

@kernel function propagate_scalar_wave_kernel!(electric, electric_dz, E_state,
                                               kernel, ::Val{forward}) where {forward}
    s = forward ? 1 : -1
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(electric)[3:end])
        a = _get_val(kernel.a, I, J)
        exp_a_p = _get_val(kernel.exp_a_p, I, J)
        exp_a_m = _get_val(kernel.exp_a_m, I, J)
        E_val = electric[I, J]
        dE_val = electric_dz[I, J]
        E2 = E_state[I, J]
        E1 = 0.5 * (E_val + s * dE_val / a)
        E1 *= exp_a_p
        E2 *= exp_a_m
        electric[I,J] = nrm_f * (E1 + E2)
        electric_dz[I,J] = nrm_f * a * s * (E1 - E2)
        E_state[I, J] = E1
    end
end

function propagate!(u::ScalarWaveField, state, p::ScalarWavePropagator)
    backend = get_backend(u.electric)
    propagate_scalar_wave_kernel!(backend)(
        u.electric, u.electric_dz, state.E_state, p.kernel, Val(true);
        ndrange = size(u.electric)[1:2])
    u
end

function inverse_propagate!(u::ScalarWaveField, state, p::ScalarWavePropagator)
    backend = get_backend(u.electric)
    propagate_scalar_wave_kernel!(backend)(
        u.electric, u.electric_dz, state.E_state, p.kernel, Val(false);
        ndrange = size(u.electric)[1:2])
    u
end
