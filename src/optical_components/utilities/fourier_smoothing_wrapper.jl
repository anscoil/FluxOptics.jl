"""
    FourierSmoothingWrapper(component, ns, ds, f)

Wrap a component to apply Fourier-space smoothing to its trainable data before
each forward pass.

Regularizes optimization by filtering high-frequency components of the trainable
parameters (e.g. a surface height map or phase mask) in Fourier space. The filter
is evaluated once at construction time and applied in-place at each forward pass.

# Arguments
- `component`: Trainable component to wrap (e.g. `FS_WPM`, `Phase`, etc.)
- `ns`: Spatial dimensions of the trainable data, e.g. `(256, 256)`
- `ds`: Spatial sampling intervals, e.g. `(2.0, 2.0)` in µm
- `f`: Filter function in Fourier space, taking frequency coordinates as arguments.
  For a 2D filter: `f(kx, ky) -> value`. Evaluated on the physical frequency grid
  defined by `ns` and `ds`.

# Use Case
Pixel-wise optimization of surface height maps or phase masks can be unstable,
with adjacent pixels diverging independently. Fourier smoothing suppresses
high-frequency instabilities by attenuating the corresponding frequency components
before each evaluation of the forward model.

# Examples
```julia
u = ScalarField(ones(ComplexF32, 256, 256), (2.0, 2.0), 1.55)
surface = FS_WPM(u, 10.0, 0.5, 1.0, 1.5; trainable=true, buffered=true)

# Biharmonic smoothing filter — suppresses high spatial frequencies
α = 1e-4  # in µm⁴
biharmonic = (kx, ky) -> 1 / (1 + 2α * (kx^2 + ky^2)^2)
wrapper = FourierSmoothingWrapper(surface, (256, 256), (2.0, 2.0), biharmonic)

# Gaussian smoothing filter with σ = 0.1 µm⁻¹ in frequency space
gaussian = (kx, ky) -> exp(-(kx^2 + ky^2) / (2 * 0.1^2))
wrapper = FourierSmoothingWrapper(surface, (256, 256), (2.0, 2.0), gaussian)
```

**Note:** The filter function operates on physical frequencies (cycles/µm), not
normalized frequencies. The `1/prod(ns)` normalization factor is baked into the
stored filter to avoid an extra scaling pass at runtime.

See also: [`BasisProjectionWrapper`](@ref), [`FS_WPM`](@ref)
"""
struct FourierSmoothingWrapper{M, C, D, B, F, P} <: AbstractPureComponent{M}
    wrapped_component::C
    mapped_data::D
    buffer::B
    filter::F
    p_f::P
    ∂p::Union{Nothing, @NamedTuple{buffer::B}}
    
    function FourierSmoothingWrapper(wrapped_component::C,
                                     mapped_data::D,
                                     buffer::B,
                                     filter::F,
                                     p_f::P,
                                     ∂p::Union{Nothing, @NamedTuple{buffer::B}}) where {C, D, B, F, P}
        M = isnothing(∂p) ? Trainable{Unbuffered} : Trainable{Buffered}
        new{M, C, D, B, F, P}(wrapped_component, mapped_data, buffer, filter, p_f, ∂p)
    end

    function FourierSmoothingWrapper(wrapped_component::C,
                                     ns::NTuple{Nd, Integer},
                                     ds::NTuple{Nd, Real},
                                     f::Function) where {M <: Trainability,
                                                         C <: AbstractPipeComponent{M},
                                                         Nd}
        mapped_data = get_data(wrapped_component)
        @assert size(mapped_data)[1:Nd] == ns "Spatial dimensions $(size(data)[1:Nd]) don't match ns=$ns"
        D = typeof(mapped_data)
        F = similar(D, complex, Nd)
        filter = F(function_to_array(f, ns, ds, true) ./ prod(ns))
        buffer = similar(mapped_data, complex(eltype(mapped_data)))
        copyto!(buffer, mapped_data)
        B = typeof(buffer)
        p_f = make_fft_plans(buffer, Tuple(1:Nd); normalize=false)
        P = typeof(p_f)
        ∂p = M == Trainable{Buffered} ? (; buffer = similar(buffer)) : nothing
        new{M, C, D, B, F, P}(wrapped_component, mapped_data, buffer, filter, p_f, ∂p)
    end
end

Functors.@functor FourierSmoothingWrapper (buffer,)

get_data(p::FourierSmoothingWrapper) = p.buffer

trainable(p::FourierSmoothingWrapper{<:Trainable}) = (; buffer = p.buffer)

function apply_smoothing!(p::FourierSmoothingWrapper)
    p.p_f.ft * p.buffer
    p.buffer .*= p.filter
    p.p_f.ift * p.buffer
    copyto!(p.mapped_data, real.(p.buffer))
    p.wrapped_component
end

function propagate!(u::ScalarField, p::FourierSmoothingWrapper)
    wrapped_component = apply_smoothing!(p)
    propagate!(u, wrapped_component)
end

propagate(u::ScalarField, p::FourierSmoothingWrapper) = propagate!(copy(u), p)
