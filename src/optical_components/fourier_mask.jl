function FourierMask(u::ScalarField{U, Nd},
        ds::NTuple{Nd, Real},
        f::Union{Function, AbstractArray} = (_...) -> 1;
        trainable::Bool = false, buffered::Bool = false
) where {Nd, U}
    if isa(f, Function)
        ns = size(u)[1:Nd]
        f = function_to_array(f, ns, ds, true)
    end
    mask = Mask(u, ds, f; trainable, buffered)
    FourierWrapper(u, phase)
end

function FourierMask(u::ScalarField{U, Nd},
        f::Union{Function, AbstractArray} = (_...) -> 1;
        trainable::Bool = false, buffered::Bool = false
) where {Nd, U}
    if isa(f, Function)
        ns = size(u)[1:Nd]
        f = function_to_array(f, ns, u.ds, true)
    end
    mask = Mask(u, f; trainable, buffered)
    FourierWrapper(u, phase)
end
