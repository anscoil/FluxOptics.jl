"""
    GainSheet(u::ScalarField, ds::NTuple, dz, Isat, f; trainable=false)
    GainSheet(u::ScalarField, dz, Isat, f; trainable=false)

Create a saturable gain sheet component.

Models thin gain medium with spatially-varying gain coefficient and saturation.
Gain follows: `g(u) = g₀ / (1 + I/Isat)` where I = |u|² is local intensity.

# Arguments
- `u::ScalarField`: Field template
- `ds::NTuple`: Spatial sampling (defaults to `u.ds`)
- `dz::Real`: Effective gain sheet thickness
- `Isat::Real`: Saturation intensity
- `f::Function`: Gain coefficient function `(x, y) -> g₀`
- `trainable::Bool`: Optimize gain profile (default: false)

# Physics

Transmission: `u_out = u_in × exp(g(I) × dz)`

where:
- `g(u) = g₀ / (1 + I/Isat)`: Gain coefficient with saturation
- `I = |u|²`: Local intensity
- High intensity → gain saturation (g → 0)
- Low intensity → linear gain (g ≈ g₀)

# Examples
```julia
u = ScalarField(ones(ComplexF64, 256, 256), (2.0, 2.0), 1.064)

# Uniform gain
gain = GainSheet(u, (2.0, 2.0), 0.1, 1e6, (x, y) -> 2.0)

# Spatially-varying gain (e.g., pumped region)
gain_region = GainSheet(u, 0.1, 1e6, (x, y) -> begin
    r = sqrt(x^2 + y^2)
    r < 50.0 ? 2.0 : 0.0
end)

# Trainable gain profile
gain_opt = GainSheet(u, 0.1, 1e6, (x, y) -> 1.0; trainable=true)
```

See also: [`Phase`](@ref), [`Mask`](@ref)
"""
struct GainSheet{M, T, A} <: AbstractPureComponent{M}
    g0::A
    dz::T
    Isat::T

    function GainSheet(g0::A, dz::T, Isat::T) where {T, A}
        new{Trainable, T, A}(g0, dz, Isat)
    end

    function GainSheet(u::ScalarField{U, Nd},
                       ds::NTuple{Nd, Real},
                       dz::Real,
                       Isat::Real,
                       f::Function;
                       trainable::Bool = false) where {Nd, T,
                                                       U <: AbstractArray{Complex{T}}}
        ns = size(u)[1:Nd]
        A = similar(U, real, Nd)
        xs = spatial_vectors(ns, ds)
        g0 = Nd == 2 ? A(f.(xs[1], xs[2]')) : A(f.(xs[1]))
        M = trainable ? Trainable : Static
        new{M, T, A}(g0, dz, Isat)
    end

    function GainSheet(u::ScalarField{U, Nd},
                       dz::Real,
                       Isat::Real,
                       f::Function;
                       trainable::Bool = false) where {Nd, T,
                                                       U <: AbstractArray{Complex{T}}}
        GainSheet(u, Tuple(u.ds), dz, Isat, f; trainable)
    end
end

Functors.@functor GainSheet (g0,)

get_data(p::GainSheet) = p.g0

trainable(p::GainSheet{<:Trainable}) = (; g0 = p.g0)

function propagate(u::ScalarField, p::GainSheet, ::Type{<:Direction})
    data = u.electric .* exp.((p.g0*p.dz) ./ (1 .+ intensity(u)/p.Isat))
    set_field_data(u, data)
end
