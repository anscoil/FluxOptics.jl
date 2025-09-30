struct FourierPhase{M, C} <: AbstractSequence{M}
    optical_components::C

    function FourierPhase(optical_components::C) where {C}
        new{Trainable, C}(optical_components)
    end

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
        wrapper = FourierWrapper(u, phase)
        M = get_trainability(wrapper)
        optical_components = get_sequence(wrapper)
        C = typeof(optical_components)
        new{M, C}(optical_components)
    end

    function FourierPhase(u::ScalarField{U, Nd},
                          f::Union{Function, AbstractArray{<:Real}} = (_...) -> 0;
                          trainable::Bool = false,
                          buffered::Bool = false) where {Nd, U}
        FourierPhase(u, Tuple(u.ds), f; trainable, buffered)
    end
end

Functors.@functor FourierPhase (optical_components,)

get_sequence(p::FourierPhase) = p.optical_components
