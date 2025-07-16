using Flux
using Functors
using AbstractFFTs

function (p::AbstractOpticalComponent)(u)
    propagate!(u, p; direction = Forward)
end

Functors.@leaf AbstractFFTs.Plan

Flux.@layer ASProp{Static} trainable=()
Flux.@layer RSProp{Static} trainable=()

Flux.@layer Phase
Flux.trainable(phi::Phase{Static}) = NamedTuple{}()
Flux.trainable(phi::Phase{Trainable}) = (; ϕ = P.ϕ)
