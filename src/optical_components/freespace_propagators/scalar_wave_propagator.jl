struct BidirectionalKernel{K}
    a::K
    exp_a_p::K
    exp_a_m::K
end

Adapt.@adapt_structure BidirectionalKernel

function BidirectionalKernel(u::ScalarWaveField, z::Real, n0::Number;
                             conjugate::Bool = false)
    a = im * compute_kz(u, n0)
    exp_a_p = @. exp(a * z)
    exp_a_m = conjugate ? (@. conj(exp_a_p)) : (@. exp(-a * z))
    BidirectionalKernel(a, exp_a_p, exp_a_m)
end

struct ScalarWavePropagator{M, K, T}  <: AbstractBidirectionalComponent{M}
    trainability::Val{M}
    z::T
    n0::Complex{T}
    kernel::BidirectionalKernel{K}
    conjugate::Bool
end

function ScalarWavePropagator(u::ScalarWaveField{U}, z::Real, n0::Number;
                              conjugate::Bool = false) where {T <: Real, U <: AbstractArray{Complex{T}}}
    z = T(z)
    n0 = Complex{T}(n0)
    kernel = BidirectionalKernel(u, z, n0; conjugate)
    ScalarWavePropagator(Val(Static), z, n0, kernel, conjugate)
end

get_n0(p::ScalarWavePropagator) = p.n0

function alloc_fp_state(u::ScalarWaveField, p::ScalarWavePropagator)
    p.conjugate ? (; E_state = nothing) : (; E_state = similar(u.electric))
end

@kernel function propagate_scalar_wave_kernel!(electric, electric_dz, E_state,
                                               kernel, ::Val{forward}) where {forward}
    s = forward ? 1 : -1
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(electric)[3:end])
        a = _get_val(kernel.a, I, J)
        exp_a_p = _get_val(kernel.exp_a_p, I, J)
        exp_a_m = _get_val(kernel.exp_a_m, I, J)
        E_val = electric[I,J]
        dE_val = electric_dz[I,J]
        E1 = 0.5 * (E_val + s * dE_val / a)
        E2 = isnothing(E_state) ? 0.5 * (E_val - s * dE_val / a) : E_state[I,J]
        E1 *= exp_a_p
        E2 *= exp_a_m
        electric[I,J] = E1 + E2
        electric_dz[I,J] = a * s * (E1 - E2)
        if !isnothing(E_state)
            E_state[I,J] = E1
        end
    end
end

@kernel function propagate_scalar_wave_adjoint_kernel!(∂electric, ∂electric_dz, ∂E_state,
                                                       kernel, ::Val{forward}
                                                       ) where {forward}
    s = forward ? 1 : -1
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(∂electric)[3:end])
        a = _get_val(kernel.a, I, J)
        exp_a_p = _get_val(kernel.exp_a_p, I, J)
        exp_a_m = _get_val(kernel.exp_a_m, I, J)
        ∂E_val = ∂electric[I,J]
        ∂dE_val = ∂electric_dz[I,J]
        ∂E1 = isnothing(∂E_state) ? 0 : ∂E_state[I,J]
        ∂E1 += ∂E_val + s * conj(a) * ∂dE_val
        ∂E2 = ∂E_val - s * conj(a) * ∂dE_val
        ∂E1 *= conj(exp_a_p)
        ∂E2 *= conj(exp_a_m)
        ∂electric[I,J] = 0.5 * ∂E1
        ∂electric_dz[I,J] = 0.5 / conj(a) * s * ∂E1
        if !isnothing(∂E_state)
            ∂E_state[I,J] = ∂E2
        end
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

function propagate_adjoint!(u::ScalarWaveField, ::Nothing,
                            state, ::Nothing,
                            p::ScalarWavePropagator)
    backend = get_backend(u.electric)
    propagate_scalar_wave_adjoint_kernel!(backend)(
        u.electric, u.electric_dz, state.E_state, p.kernel, Val(true);
        ndrange = size(u.electric)[1:2])
    u
end

function inverse_propagate_adjoint!(u::ScalarWaveField, ::Nothing,
                                    state, ::Nothing,
                                    p::ScalarWavePropagator)
    backend = get_backend(u.electric)
    propagate_scalar_wave_adjoint_kernel!(backend)(
        u.electric, u.electric_dz, state.E_state, p.kernel, Val(false);
        ndrange = size(u.electric)[1:2])
    u
end
