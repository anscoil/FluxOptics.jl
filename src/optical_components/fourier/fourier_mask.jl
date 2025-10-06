"""
    FourierMask(u::ScalarField, f; trainable=false, buffered=false)

Apply amplitude/complex mask in Fourier domain.

Convenient constructor for `FourierWrapper(u, Mask(...))`. The function f
receives spatial frequencies (fx, fy) as arguments.

# Arguments
- `u::ScalarField`: Field template
- `f`: Mask function `(fx, fy) -> m` in Fourier space (complex)
- `trainable::Bool`: Enable optimization (default: false)
- `buffered::Bool`: Pre-allocate gradients (default: false)

# Examples
```julia
u = ScalarField(ones(ComplexF64, 256, 256), (1.0, 1.0), 1.064)

# Low-pass filter (sharp cutoff)
f_cutoff = 0.2  # 1/μm
lowpass = FourierMask(u, (fx, fy) -> sqrt(fx^2 + fy^2) < f_cutoff ? 1.0 : 0.0)

# Gaussian filter
sigma_f = 0.15
gaussian_filter = FourierMask(u, (fx, fy) -> exp(-(fx^2 + fy^2)/(2*sigma_f^2)))
```

See also: [`FourierPhase`](@ref), [`FourierWrapper`](@ref), [`Mask`](@ref)
"""
struct FourierMask{M, C} <: AbstractSequence{M}
    optical_components::C

    function FourierMask(optical_components::C) where {N,
                                                       C <:
                                                       NTuple{N, AbstractPipeComponent}}
        new{Trainable, C}(optical_components)
    end

    function FourierMask(u::ScalarField{U, Nd},
                         ds::NTuple{Nd, Real},
                         f::Union{Function, AbstractArray} = (_...) -> 1;
                         trainable::Bool = false,
                         buffered::Bool = false) where {Nd, U}
        if isa(f, Function)
            ns = size(u)[1:Nd]
            f = function_to_array(f, ns, ds, true)
        end
        mask = Mask(u, ds, f; trainable, buffered)
        wrapper = FourierWrapper(u, mask)
        M = get_trainability(wrapper)
        optical_components = get_sequence(wrapper)
        C = typeof(optical_components)
        new{M, C}(optical_components)
    end

    function FourierMask(u::ScalarField{U, Nd},
                         f::Union{Function, AbstractArray} = (_...) -> 1;
                         trainable::Bool = false,
                         buffered::Bool = false) where {Nd, U}
        FourierMask(u, Tuple(u.ds), f; trainable, buffered)
    end
end

Functors.@functor FourierMask (optical_components,)

get_sequence(p::FourierMask) = p.optical_components
