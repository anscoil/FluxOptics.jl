"""
    ScalarSource(u::ScalarField; trainable=false, buffered=false)

Create a scalar optical source from a field template.

Sources generate optical fields at the beginning of an optical system. The source
stores a copy of the input field and reproduces it (or an optimized version if trainable)
when propagated.

# Arguments
- `u::ScalarField`: Template field defining spatial grid, wavelength, and initial amplitude
- `trainable::Bool`: If true, field amplitude/phase can be optimized (default: false)
- `buffered::Bool`: If true, pre-allocate gradient buffers (default: false)

# Examples
```jldoctest
julia> xv, yv = spatial_vectors(64, 64, 2.0, 2.0);

julia> w0 = 10.0;

julia> u0 = ScalarField(Gaussian(w0)(xv, yv), (2.0, 2.0), 1.064);

julia> source = ScalarSource(u0); # Static source (fixed beam profile)

julia> u = propagate(source);

julia> source_train = ScalarSource(u0; trainable=true, buffered=true); # Trainable source (optimizable beam)

julia> phase_mask = Phase(u, (x, y) -> 0.01 * (x^2 + y^2));

julia> propagator = ASProp(u0, 500.0);

julia> system = source_train |> phase_mask |> propagator;

```

See also: [`get_source`](@ref), [`propagate`](@ref)
"""
struct ScalarSource{M, S} <: AbstractCustomSource{M}
    u0::S
    uf::S
    ∂p::Union{Nothing, @NamedTuple{u0::S}}

    function ScalarSource(u0::S, uf::S, ∂p) where {S}
        new{Static, S}(u0, uf, ∂p)
    end

    function ScalarSource(u::S;
                          trainable::Bool = false,
                          buffered::Bool = false) where {U <: AbstractArray{<:Complex},
                                                         S <: ScalarField{U}}
        u0 = copy(u)
        uf = similar(u)
        M = trainability(trainable, buffered)
        ∂p = (trainable && buffered) ? (; u0 = similar(u0)) : nothing
        new{M, S}(u0, uf, ∂p)
    end
end

Functors.@functor ScalarSource (u0,)

Base.collect(p::ScalarSource) = collect(p.u0)
Base.size(p::ScalarSource) = size(p.u0)

trainable(p::ScalarSource{<:Trainable}) = (; u0 = p.u0)

get_preallocated_gradient(p::ScalarSource{Trainable{Buffered}}) = p.∂p

function propagate(p::ScalarSource)
    copyto!(p.uf.electric, p.u0.electric)
    p.uf
end

propagate_and_save(p::ScalarSource) = propagate(p)

function backpropagate_with_gradient(∂v, ∂p::NamedTuple, p::ScalarSource{<:Trainable})
    copyto!(∂p.u0.electric, ∂v.electric)
    ∂p
end

get_data(p::ScalarSource) = p.u0.electric

function Base.fill!(p::ScalarSource, u0::ScalarField)
    copyto!(p.u0.electric, u0.electric)
end

"""
    get_source(source::ScalarSource)

Access the current field stored in a scalar source.

Returns a copy of the source's internal field. Useful for inspection or
extracting the optimized beam profile after training.

# Examples
```julia
source = ScalarSource(u0; trainable=true)
# ... after optimization ...
optimized_field = get_source(source)
```

See also: [`ScalarSource`](@ref)
"""
function get_source(p::ScalarSource)
    p.u0
end
