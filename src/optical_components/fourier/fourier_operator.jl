"""
    FourierOperator(u::ScalarField, direct::Bool)

Create a Fourier transform operator (FFT or IFFT).

Low-level component for manual Fourier domain operations. Applies either
forward FFT (`direct=true`) or inverse IFFT (`direct=false`) to the field.
Most users should use `FourierWrapper`, `FourierPhase`, or `FourierMask` instead.

# Arguments
- `u::ScalarField`: Field template (defines grid size and dimensions)
- `direct::Bool`: `true` for FFT, `false` for IFFT

# Direction Behavior
- **Forward direction**: Applies FFT if `direct=true`, IFFT if `direct=false`
- **Backward direction**: Reversed (IFFT if `direct=true`, FFT if `direct=false`)

# Examples
```julia
u = ScalarField(ones(ComplexF64, 256, 256), (1.0, 1.0), 1.064)

# Forward FFT
fft_op = FourierOperator(u, true)
u_freq = propagate(u, fft_op, Forward)

# Inverse FFT
ifft_op = FourierOperator(u, false)
u_back = propagate(u_freq, ifft_op, Forward)
```

**Note:** `FourierOperator` is used internally by `FourierWrapper` to create
FFT → component → IFFT sequences. For most use cases, prefer the higher-level
wrappers.

See also: [`FourierWrapper`](@ref), [`FourierPhase`](@ref), [`FourierMask`](@ref)
"""
struct FourierOperator{M, S, P} <: AbstractPureComponent{M}
    p_f::P
    s::S
    direct::Bool

    function FourierOperator(p_f::P, s::S, direct::Bool) where {S, P}
        new{Static, S, P}(p_f, s, direct)
    end

    function FourierOperator(p_f::FFTPlans, direct::Bool)
        s = size(p_f.ft)
        @assert s == size(p_f.ift)
        d = fftdims(p_f.ft)
        @assert d == fftdims(p_f.ift)
        P = typeof(p_f)
        S = Val{(s, d)}
        new{Static, S, P}(p_f, S(), direct)
    end

    function FourierOperator(u::ScalarField{U, Nd}, direct::Bool) where {Nd, U}
        u_plan = similar(u.electric)
        p_f = make_fft_plans(u_plan, Tuple(1:Nd))
        FourierOperator(p_f, direct)
    end
end

get_data(p::FourierOperator) = ()

function propagate!(u::ScalarField, p::FourierOperator, ::Type{Forward})
    if p.direct
        compute_ft!(p.p_f, u)
    else
        compute_ift!(p.p_f, u)
    end
end

function propagate!(u::ScalarField, p::FourierOperator, ::Type{Backward})
    if !p.direct
        compute_ft!(p.p_f, u)
    else
        compute_ift!(p.p_f, u)
    end
end

function propagate(u::ScalarField, p::FourierOperator, direction::Type{<:Direction})
    propagate!(copy(u), p, direction)
end
