"""
    Phase(u::ScalarField, f; trainable=false, buffered=false)
    Phase(u::ScalarField, ds::NTuple, f; trainable=false, buffered=false)

Create a phase mask component.

Applies a spatially-varying phase shift to an optical field: `u → u × exp(iφ(x,y))`.
The phase can be specified as a function or an array, and can be made trainable
for inverse design optimization.

# Arguments
- `u::ScalarField`: Field template (defines grid size and type)
- `ds::NTuple`: Spatial sampling (defaults to `u.ds`)
- `f`: Phase function `(x, y) -> φ` or phase array
- `trainable::Bool`: Enable gradient-based optimization (default: false)
- `buffered::Bool`: Pre-allocate gradient buffers for performance (default: false)

# Examples
```julia
u = ScalarField(ones(ComplexF64, 256, 256), (2.0, 2.0), 1.064)

# Parabolic phase (lens-like)
phase_lens = Phase(u, (x, y) -> π/(1000^2) * (x^2 + y^2))

# Trainable phase mask (starts at zero)
phase_opt = Phase(u, (x, y) -> 0.0; trainable=true, buffered=true)

# From measured/computed array
phase_data = load_measurement(...)
phase_measured = Phase(u, phase_data)

# Use in system
system = ScalarSource(u) |> phase_opt |> ASProp(u, 1000.0)
```

See also: [`Mask`](@ref), [`TeaDOE`](@ref), [`FourierPhase`](@ref)
"""
struct Phase{M, A, U} <: AbstractCustomComponent{M}
    ϕ::A
    ∂p::Union{Nothing, @NamedTuple{ϕ::A}}
    u::Union{Nothing, U}

    function Phase(ϕ::A, ∂p::Union{Nothing, @NamedTuple{ϕ::A}}, u::U) where {A, U}
        M = isnothing(u) ? Trainable{Unbuffered} : Trainable{Buffered}
        new{M, A, U}(ϕ, ∂p, u)
    end

    function Phase(u::ScalarField{U, Nd},
                   ds::NTuple{Nd, Real},
                   f::Union{Function, AbstractArray{<:Real}} = (_...) -> 0;
                   trainable::Bool = false,
                   buffered::Bool = false) where {Nd, U}
        M = trainability(trainable, buffered)
        @assert Nd in (1, 2)
        if isa(f, Function)
            A = similar(U, real, Nd)
            ns = size(u)[1:Nd]
            ϕ = A(function_to_array(f, ns, ds))
        else
            @assert isbroadcastable(f, u)
            A = similar(U, real, ndims(f))
            ϕ = A(f)
        end
        ∂p = (trainable && buffered) ? (; ϕ = similar(ϕ)) : nothing
        u = (trainable && buffered) ? similar(u.electric) : nothing
        A = typeof(ϕ)
        new{M, A, U}(ϕ, ∂p, u)
    end

    function Phase(u::ScalarField{U, Nd},
                   f::Union{Function, AbstractArray{<:Real}} = (_...) -> 0;
                   trainable::Bool = false,
                   buffered::Bool = false) where {Nd, U}
        Phase(u, Tuple(u.ds), f; trainable, buffered)
    end
end

Functors.@functor Phase (ϕ,)

get_data(p::Phase) = p.ϕ

trainable(p::Phase{<:Trainable}) = (; ϕ = p.ϕ)

get_preallocated_gradient(p::Phase{Trainable{Buffered}}) = p.∂p

alloc_saved_buffer(u::ScalarField, p::Phase{Trainable{Unbuffered}}) = similar(u.electric)

get_saved_buffer(p::Phase{Trainable{Buffered}}) = p.u

function propagate!(u::ScalarField, p::Phase, direction::Type{<:Direction})
    s = sign(direction)
    @. u.electric *= cis(s*p.ϕ)
    u
end

function backpropagate!(u::ScalarField, p::Phase, direction::Type{<:Direction})
    propagate!(u, p, reverse(direction))
end

function propagate_and_save!(u::ScalarField,
                             p::Phase{Trainable{Buffered}},
                             direction::Type{<:Direction})
    copyto!(p.u, u.electric)
    propagate!(u, p, direction)
end

function propagate_and_save!(u::ScalarField,
                             u_saved::AbstractArray,
                             p::Phase{Trainable{Unbuffered}},
                             direction::Type{<:Direction})
    copyto!(u_saved, u.electric)
    propagate!(u, p, direction)
end

function compute_phase_gradient!(∂ϕ::AbstractArray{<:Real, Nd},
                                 u_saved,
                                 ∂u::ScalarField,
                                 direction) where {Nd}
    sdims = (Nd + 1):ndims(∂u.electric)
    s = sign(direction)
    g = @. s*imag(∂u.electric*conj(u_saved))
    copyto!(∂ϕ, sum(g; dims = sdims))
end

function compute_phase_gradient!(∂ϕ::Array{<:Real, Nd},
                                 u_saved,
                                 ∂u::ScalarField,
                                 direction) where {Nd}
    sdims = 3:ndims(∂u)
    s = sign(direction)
    ∂ϕ .= 0
    @inbounds for idx in CartesianIndices(size(∂u)[sdims])
        @inbounds for j in axes(∂ϕ, 2), i in axes(∂ϕ, 1)
            full_idx = (i, j, Tuple(idx)...)
            val = imag(∂u.electric[full_idx...] * conj(u_saved[full_idx...]))
            ∂ϕ[i, j] += s*val
        end
    end
    ∂ϕ
end

function backpropagate_with_gradient!(∂v::ScalarField,
                                      u_saved::AbstractArray,
                                      ∂p::NamedTuple,
                                      p::Phase{<:Trainable},
                                      direction::Type{<:Direction})
    ∂u = backpropagate!(∂v, p, direction)
    compute_phase_gradient!(∂p.ϕ, u_saved, ∂u, direction)
    (∂u, ∂p)
end
