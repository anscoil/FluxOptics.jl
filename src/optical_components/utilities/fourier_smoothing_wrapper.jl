"""
    FourierSmoothingWrapper(component, ns, ds, f)

Wrap a component to apply Fourier-space smoothing to its trainable data before
each forward pass.

Regularizes optimization by filtering high-frequency components of the trainable
parameters (e.g. a surface height map or phase mask) in Fourier space. The filter
is evaluated once at construction time and applied in-place at each forward pass.

# Arguments
- `component`: Trainable component to wrap (e.g. `FS_WPM`, `Phase`, etc.)
- `ns`: Spatial dimensions of the trainable data
- `ds`: Spatial sampling intervals
- `f`: Filter function in Fourier space. For a 2D filter, signature is `f(fx, fy)`
  where `fx`, `fy` are spatial frequencies in cycles per unit of length.
  Evaluated on the physical frequency grid defined by `ns` and `ds`.

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
α = 1e-4
biharmonic = (fx, fy) -> 1 / (1 + 2α * (fx^2 + fy^2)^2)
wrapper = FourierSmoothingWrapper(surface, (256, 256), (2.0, 2.0), biharmonic)

# Gaussian smoothing filter
gaussian = (fx, fy) -> exp(-(fx^2 + fy^2) / (2 * 0.1^2))
wrapper = FourierSmoothingWrapper(surface, (256, 256), (2.0, 2.0), gaussian)
```

See also: [`BasisProjectionWrapper`](@ref), [`FS_WPM`](@ref)
"""
struct FourierSmoothingWrapper{M, C, D, AD, B, F, P} <: AbstractPureComponent{M}
    trainability::Val{M}
    wrapped_component::C
    mapped_data::D
    aux_data::AD
    buffer::B
    filter::F
    p_f::P
    ∂p::Union{Nothing, @NamedTuple{buffer::B}}
end

Functors.@functor FourierSmoothingWrapper (buffer, aux_data)

function FourierSmoothingWrapper(wrapped_component::C,
                                 ns::NTuple{Nd, Integer},
                                 ds::NTuple{Nd, Real},
                                 f::Function) where {M <: Trainability,
                                                     C <: AbstractPipeComponent{M}, Nd}
    mapped_data = get_data(wrapped_component)
    @assert size(mapped_data)[1:Nd] == ns "Spatial dimensions $(size(data)[1:Nd]) don't match ns=$ns"
    D = typeof(mapped_data)
    F = similar(D, complex, Nd)
    filter = F(function_to_array(f, ns, ds, true) ./ prod(ns))
    buffer = similar(mapped_data, complex(eltype(mapped_data)))
    copyto!(buffer, mapped_data)
    p_f = make_fft_plans(buffer, Tuple(1:Nd); normalize=false)
    aux_data = auxiliary_trainable(wrapped_component)
    ∂p = M == Trainable{Buffered} ? (; buffer = similar(buffer)) : nothing
    FourierSmoothingWrapper(Val(M), wrapped_component, mapped_data, aux_data, buffer, filter, p_f, ∂p)
end

data_symbol_chain(p::FourierSmoothingWrapper) = (:buffer,)

function trainable(p::FourierSmoothingWrapper{<:Trainable})
    (; buffer = p.buffer, aux_data = p.aux_data)
end

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
