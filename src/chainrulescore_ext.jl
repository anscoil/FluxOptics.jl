using ChainRulesCore

function ChainRulesCore.rrule(
        ::typeof(propagate!), u, p::AbstractOpticalComponent{Static};
        direction::Type{<:Direction} = Forward)
    v = propagate!(u, p; direction = direction)

    function pullback(Δv)
        ∂u = propagate!(Δv, p; direction = (direction == Forward ? Backward : Forward))
        return (NoTangent(), ∂u, NoTangent())
    end

    return v, pullback
end

function ChainRulesCore.rrule(
        ::typeof(propagate!), u, phi::Phase{Trainable};
        direction::Type{<:Direction} = Forward)
    v = propagate!(u, phi; direction = direction, save_u = true)
    function pullback(Δv)
        ∂u = propagate!(Δv, phi; direction = (direction == Forward ? Backward : Forward))
        sdims = Tuple(3:ndims(Δv))
        @views phi.∇ϕ .= dropdims(
            sum(imag.(∂u .* conj.(phi.u_fwd)), dims = sdims), dims = sdims)
        return (NoTangent(), ∂u, Tangent{Phase}(ϕ = phi.∇ϕ))
    end

    return v, pullback
end
