struct FourierMask{M, C} <: AbstractSequence{M}
    optical_components::OpticalSequence{M, C}

    function FourierMask(optical_components::C) where {C}
        new{Trainable, C}(optical_components)
    end

    function FourierMask(u::ScalarField{U, Nd},
                         ds::NTuple{Nd, Real},
                         f::Union{Function, AbstractArray} = (_...) -> 1;
                         trainable::Bool = false,
                         buffered::Bool = false) where {Nd, U}
        if isa(f, Function)
            ns = size(u)[1:Nd]
            f = function_to_array(f, ns, ds, true)
        end
        mask = Mask(u, ds, f; trainable, buffered)
        wrapper = FourierWrapper(u, mask)
        M = get_trainability(wrapper)
        optical_components = get_sequence(wrapper)
        C = typeof(optical_components)
        new{M, C}(optical_components)
    end

    function FourierMask(u::ScalarField{U, Nd},
                         f::Union{Function, AbstractArray} = (_...) -> 1;
                         trainable::Bool = false,
                         buffered::Bool = false) where {Nd, U}
        FourierMask(u, u.ds, f; trainable, buffered)
    end
end

Functors.@functor FourierMask (optical_components,)

get_sequence(p::FourierMask) = p.optical_components
