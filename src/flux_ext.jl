using Flux
using Functors
using AbstractFFTs

function (p::AbstractOpticalComponent)(u)
    propagate!(u, p, Forward)
end

Flux.trainable(p::AbstractOpticalComponent) = OpticalComponents.trainable(p)

Functors.@leaf AbstractFFTs.Plan

Flux.@layer ASProp{Static}
Flux.@layer RSProp{Static}

Flux.@layer Phase
