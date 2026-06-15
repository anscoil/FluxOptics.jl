"""
    FourierOperator(u::ScalarField, direct::Bool)

Create a Fourier transform operator (FFT or IFFT).

Low-level component for manual Fourier domain operations. Applies either
forward FFT (`direct=true`) or inverse IFFT (`direct=false`) to the field.
Most users should use `FourierWrapper`, `FourierPhase`, or `FourierMask` instead.

# Arguments
- `u::ScalarField`: Field template (defines grid size and dimensions)
- `direct::Bool`: `true` for FFT, `false` for IFFT

# Examples
```julia
u = ScalarField(ones(ComplexF64, 256, 256), (1.0, 1.0), 1.064)

# Forward FFT
fft_op = FourierOperator(u, true)
u_freq = propagate(u, fft_op)

# Inverse FFT
ifft_op = FourierOperator(u, false)
u_back = propagate(u_freq, ifft_op)
```

**Note:** `FourierOperator` is used internally by `FourierWrapper` to create
FFT → component → IFFT sequences. For most use cases, prefer the higher-level
wrappers.

See also: [`FourierWrapper`](@ref), [`FourierPhase`](@ref), [`FourierMask`](@ref)
"""
struct FourierOperator{S, T, P} <: AbstractPureComponent{Static}
    p_f::P
    nrm_f::T
    s::S
    direct::Bool
end

function FourierOperator(p_f::FFTPlans, nrm_f::Union{Nothing, Number}, direct::Bool)
    s = size(p_f.ft)
    @assert s == size(p_f.ift)
    d = fftdims(p_f.ft)
    @assert d == fftdims(p_f.ift)
    S = Val{(s, d)}
    FourierOperator(p_f, nrm_f, S(), direct)
end

function FourierOperator(u::ScalarField{U, Nd}, direct::Bool;
                         normalize::Bool = true) where {Nd, U}
    u_plan = similar(u.electric)
    p_f, nrm_f = make_fft_plans(u_plan, Tuple(1:Nd); normalize)
    FourierOperator(p_f, nrm_f, direct)
end

get_data(p::FourierOperator) = ()

function propagate!(u::AbstractField, p::FourierOperator)
    if p.direct
        compute_ft!(p.p_f, u)
    else
        compute_ift!(p.p_f, u)
    end
end

function backpropagate!(u::AbstractField, p::FourierOperator)
    if p.direct
        compute_ift!(p.p_f, u)
    else
        compute_ft!(p.p_f, u)
    end
end

propagate(u, p::FourierOperator) = propagate!(copy(u), p)
