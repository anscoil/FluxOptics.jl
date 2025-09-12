using ChainRulesCore
using Functors: fleaves
using LinearAlgebra: mul!

function ChainRulesCore.rrule(
        ::typeof(propagate), u, p::AbstractCustomComponent{Static},
        direction::Type{<:Direction})
    v = propagate(u, p, direction)

    function pullback(∂v)
        ∂u = backpropagate(∂v, p, direction)
        return (NoTangent(), ∂u, NoTangent(), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(
        ::typeof(propagate), u, p::P,
        direction::Type{<:Direction}
) where {P <: AbstractCustomComponent{Trainable{Buffered}}}
    v = propagate_and_save(u, p, direction)

    function pullback(∂v)
        ∂p = get_preallocated_gradient(p)
        u_saved = get_saved_buffer(p)
        ∂u, ∂p = backpropagate_with_gradient(∂v, u_saved, ∂p, p, direction)
        return (NoTangent(), ∂u, Tangent{P}(; ∂p...), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(
        ::typeof(propagate), u, p::P,
        direction::Type{<:Direction}
) where {P <: AbstractCustomComponent{Trainable{Unbuffered}}}
    u_saved = alloc_saved_buffer(u, p)
    v = propagate_and_save(u, u_saved, p, direction)

    function pullback(∂v)
        ∂p = alloc_gradient(p)
        ∂u, ∂p = backpropagate_with_gradient(∂v, u_saved, ∂p, p, direction)
        return (NoTangent(), ∂u, Tangent{P}(; ∂p...), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(
        ::typeof(propagate!), u, p::AbstractCustomComponent{Static},
        direction::Type{<:Direction})
    v = propagate!(u, p, direction)

    function pullback(∂v)
        ∂u = backpropagate!(∂v, p, direction)
        return (NoTangent(), ∂u, NoTangent(), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(
        ::typeof(propagate!), u, p::P,
        direction::Type{<:Direction}
) where {P <: AbstractCustomComponent{Trainable{Buffered}}}
    v = propagate_and_save!(u, p, direction)

    function pullback(∂v)
        ∂p = get_preallocated_gradient(p)
        u_saved = get_saved_buffer(p)
        ∂u, ∂p = backpropagate_with_gradient!(∂v, u_saved, ∂p, p, direction)
        return (NoTangent(), ∂u, Tangent{P}(; ∂p...), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(
        ::typeof(propagate!), u, p::P,
        direction::Type{<:Direction}
) where {P <: AbstractCustomComponent{Trainable{Unbuffered}}}
    u_saved = alloc_saved_buffer(u, p)
    v = propagate_and_save!(u, u_saved, p, direction)

    function pullback(∂v)
        ∂p = alloc_gradient(p)
        ∂u, ∂p = backpropagate_with_gradient!(∂v, u_saved, ∂p, p, direction)
        return (NoTangent(), ∂u, Tangent{P}(; ∂p...), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(
        ::typeof(propagate), p::AbstractCustomSource{Static})
    v = propagate(p)

    function pullback(∂v)
        return (NoTangent(), NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(propagate), p::P
) where {P <: AbstractCustomSource{Trainable{Buffered}}}
    v = propagate_and_save(p)

    function pullback(∂v)
        ∂p = get_preallocated_gradient(p)
        ∂p = backpropagate_with_gradient(∂v, ∂p, p)
        return (NoTangent(), Tangent{P}(; ∂p...))
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(propagate), p::P
) where {P <: AbstractCustomSource{Trainable{Unbuffered}}}
    v = propagate_and_save(p)

    function pullback(∂v)
        ∂p = alloc_gradient(p)
        ∂p = backpropagate_with_gradient(∂v, ∂p, p)
        return (NoTangent(), Tangent{P}(; ∂p...))
    end

    return v, pullback
end

function ChainRulesCore.rrule(::Type{<:ScalarField}, data, ds,
        lambdas::NamedTuple, tilts::NamedTuple)
    y = ScalarField(data, ds, lambdas, tilts)
    function pullback(∂y)
        (NoTangent(), ∂y.data, NoTangent(), NoTangent(), NoTangent())
    end
    return y, pullback
end

function ChainRulesCore.ProjectTo(u::ScalarField)
    function pullback(∂y)
        if ∂y.data isa NoTangent
            NoTangent()
        else
            ScalarField(∂y.data, u.ds, u.lambdas, u.tilts)
        end
    end
    pullback
end

function ChainRulesCore.rrule(::typeof(compute_ft!), p_f, u)
    v = compute_ft!(p_f, u)

    function pullback(∂v)
        ∂u = compute_ift!(p_f, ∂v)
        return (NoTangent(), NoTangent(), ∂u)
    end

    return v, pullback
end

function ChainRulesCore.rrule(::typeof(compute_ift!), p_f, u)
    v = compute_ift!(p_f, u)

    function pullback(∂v)
        ∂u = compute_ft!(p_f, ∂v)
        return (NoTangent(), NoTangent(), ∂u)
    end

    return v, pullback
end

function compute_basis_projection!(proj_coeffs, r_basis, r_data)
    mul!(proj_coeffs, r_basis', r_data)
end

function ChainRulesCore.rrule(::typeof(set_basis_projection!), p::P
) where {P <: BasisProjectionWrapper}
    wrapped_component = set_basis_projection!(p)

    function pullback(∂c)
        ∂mapped_data = filter(
            x -> !(x isa ChainRulesCore.ZeroTangent) &&
                 !(x isa ChainRulesCore.NoTangent) &&
                 (x isa AbstractArray) &&
                 (x isa AbstractArray),
            fleaves(∂c))[1]
        if isbuffered(p)
            ∂p = p.∂p
            mul!(∂p.proj_coeffs, p.basis', reshape(∂mapped_data, :))
        else
            ∂p = (; proj_coeffs = p.basis' * reshape(∂mapped_data, :))
        end
        return (NoTangent(), Tangent{P}(; ∂p...))
    end

    return wrapped_component, pullback
end
