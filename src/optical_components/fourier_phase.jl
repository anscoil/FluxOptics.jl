function FourierPhase(u::ScalarField{U, Nd},
        ds::NTuple{Nd, Real},
        f::Union{Function, AbstractArray{<:Real}} = (_...) -> 0;
        trainable::Bool = false, buffered::Bool = false
) where {Nd, U}
    phase = Phase(u, ds, f; trainable, buffered)
    FourierWrapper(u, phase)
end

function FourierPhase(u::ScalarField{U, Nd},
        f::Union{Function, AbstractArray{<:Real}} = (_...) -> 0;
        trainable::Bool = false, buffered::Bool = false
) where {Nd, U}
    phase = Phase(u, f; trainable, buffered)
    FourierWrapper(u, phase)
end
