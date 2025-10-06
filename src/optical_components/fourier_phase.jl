"""
    FourierPhase(u::ScalarField, f; trainable=false, buffered=false)

Apply phase mask in Fourier domain.

Convenient constructor for `FourierWrapper(u, Phase(...))`. The function f
receives spatial frequencies (fx, fy) as arguments.

# Arguments
- `u::ScalarField`: Field template
- `f`: Phase function `(fx, fy) -> φ` in Fourier space
- `trainable::Bool`: Enable optimization (default: false)
- `buffered::Bool`: Pre-allocate gradients (default: false)

# Examples
```julia
u = ScalarField(ones(ComplexF64, 256, 256), (1.0, 1.0), 1.064)

# Parabolic phase in frequency domain
fourier_phase = FourierPhase(u, (fx, fy) -> 0.01*(fx^2 + fy^2))

# Low-pass filter (soft)
sigma_f = 0.1  # 1/μm
lowpass = FourierPhase(u, (fx, fy) -> -π*(fx^2 + fy^2)/(2*sigma_f^2))
```

See also: [`FourierMask`](@ref), [`FourierWrapper`](@ref), [`Phase`](@ref)
"""
struct FourierPhase{M, C} <: AbstractSequence{M}
    optical_components::C

    function FourierPhase(optical_components::C) where {N,
                                                        C <:
                                                        NTuple{N, AbstractPipeComponent}}
        new{Trainable, C}(optical_components)
    end

    function FourierPhase(u::ScalarField{U, Nd},
                          ds::NTuple{Nd, Real},
                          f::Union{Function, AbstractArray{<:Real}} = (_...) -> 0;
                          trainable::Bool = false,
                          buffered::Bool = false) where {Nd, U}
        if isa(f, Function)
            ns = size(u)[1:Nd]
            f = function_to_array(f, ns, ds, true)
        end
        phase = Phase(u, ds, f; trainable, buffered)
        wrapper = FourierWrapper(u, phase)
        M = get_trainability(wrapper)
        optical_components = get_sequence(wrapper)
        C = typeof(optical_components)
        new{M, C}(optical_components)
    end

    function FourierPhase(u::ScalarField{U, Nd},
                          f::Union{Function, AbstractArray{<:Real}} = (_...) -> 0;
                          trainable::Bool = false,
                          buffered::Bool = false) where {Nd, U}
        FourierPhase(u, Tuple(u.ds), f; trainable, buffered)
    end
end

Functors.@functor FourierPhase (optical_components,)

get_sequence(p::FourierPhase) = p.optical_components
