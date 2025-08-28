struct FieldProbe{M} <: AbstractPureComponent{M}
    function FieldProbe()
        new{Static}()
    end
end

Functors.@functor FieldProbe ()

function propagate(u, p::FieldProbe, ::Type{<:Direction})
    (u, copy(u))
end
