function FourierWrapper(p_f::FFTPlans, wrapped_components::Vararg{AbstractPipeComponent})
    ft = FourierOperator(p_f, true)
    ift = FourierOperator(p_f, false)
    OpticalSequence(ft, wrapped_components..., ift)
end

"""
    FourierWrapper(u::ScalarField, component::AbstractPipeComponent)

Wrap a component to operate in Fourier domain.

Applies FFT before the component and IFFT after, allowing spatial-domain
components to operate on frequency content. Equivalent to:
`u → IFFT[component(FFT[u])]`

# Arguments
- `u::ScalarField`: Field template
- `component`: Component to wrap (operates on Fourier transform)

# Examples
```julia
u = ScalarField(ones(ComplexF64, 256, 256), (1.0, 1.0), 1.064)

# Apply phase mask in Fourier domain
phase_real = Phase(u, (x, y) -> 0.01*x^2)
phase_fourier = FourierWrapper(u, phase_real)

# Equivalent to frequency-domain filtering
v = propagate(u, phase_fourier, Forward)
```

See also: [`FourierPhase`](@ref), [`FourierMask`](@ref)
"""
function FourierWrapper(u::ScalarField{U, Nd},
                        wrapped_components::Vararg{AbstractPipeComponent}) where {Nd, U}
    u_plan = similar(u.electric)
    p_f = make_fft_plans(u_plan, Tuple(1:Nd))
    FourierWrapper(p_f, wrapped_components...)
end

include("fourier_phase.jl")
include("fourier_mask.jl")
