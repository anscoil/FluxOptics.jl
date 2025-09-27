function FourierPhase(u::ScalarField{U, Nd},
                      ds::NTuple{Nd, Real},
                      f::Union{Function, AbstractArray{<:Real}} = (_...) -> 0;
                      trainable::Bool = false,
                      buffered::Bool = false) where {Nd, U}
    if isa(f, Function)
        ns = size(u)[1:Nd]
        f = function_to_array(f, ns, ds, true)
    end
    phase = Phase(u, ds, f; trainable, buffered)
    FourierWrapper(u, phase)
end

function FourierPhase(u::ScalarField{U, Nd},
                      f::Union{Function, AbstractArray{<:Real}} = (_...) -> 0;
                      trainable::Bool = false,
                      buffered::Bool = false) where {Nd, U}
    if isa(f, Function)
        ns = size(u)[1:Nd]
        f = function_to_array(f, ns, u.ds, true)
    end
    phase = Phase(u, f; trainable, buffered)
    FourierWrapper(u, phase)
end
