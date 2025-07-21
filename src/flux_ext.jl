using Flux
using Functors
using AbstractFFTs

function (p::AbstractOpticalComponent)(u; direction::Type{<:Direction} = Forward)
    propagate(u, p, direction)
end

Flux.trainable(p::AbstractOpticalComponent) = OpticalComponents.trainable(p)

Functors.@leaf AbstractFFTs.Plan

Flux.@layer ASProp{Static}
Flux.@layer RSProp{Static}

Flux.@layer Phase
