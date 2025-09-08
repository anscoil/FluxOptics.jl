struct PointwiseProx{F} <: AbstractProximalOperator
    f::F
    function PointwiseProx(f::F) where {F}
        new{F}(f)
    end
end

init(prox::PointwiseProx, x::AbstractArray) = ()

function apply!(prox::PointwiseProx, state, x::AbstractArray)
    @. x = prox.f(x, state...)
end

include("ista_prox.jl")
include("clamp_prox.jl")
include("positive_prox.jl")
