simplify(p::AbstractPipeComponent) = p

# Phase

function Base.merge(p1::Phase{Static}, p2::Phase{Static})
    if size(p1.ϕ) == size(p2.ϕ)
        @. p1.ϕ += p2.ϕ
        OpticalSequence(p1)
    else
        OpticalSequence(p1, p2)
    end
end

# Mask

function Base.merge(p1::Mask{Static}, p2::Mask{Static})
    if size(p1.m) == size(p2.m)
        @. p1.m *= p2.m
        OpticalSequence(p1)
    else
        OpticalSequence(p1, p2)
    end
end

function Base.merge(p1::Mask{Static}, p2::Phase{Static})
    if size(p1.m) == size(p2.ϕ)
        @. p1.m *= cis(p2.ϕ)
        OpticalSequence(p1)
    else
        OpticalSequence(p1, p2)
    end
end

function Base.merge(p1::Phase{Static}, p2::Mask{Static})
    if size(p1.ϕ) == size(p2.m)
        @. p2.m *= cis(p1.ϕ)
        OpticalSequence(p2)
    else
        OpticalSequence(p1, p2)
    end
end

# FourierOperator

function Base.merge(p1::FourierOperator{Static, S},
                    p2::FourierOperator{Static, S}) where {S}
    if p1.direct != p2.direct
        OpticalSequence()
    else
        OpticalSequence(p1, p2)
    end
end

# OpticalSequence

function Base.merge(p1::AbstractPipeComponent, p2::AbstractPipeComponent)
    OpticalSequence(simplify(p1), simplify(p2))
end

function Base.merge(p1::OpticalSequence, p2::AbstractPipeComponent)
    merge(OpticalSequence(), OpticalSequence(p1.optical_components..., p2))
end

function Base.merge(p1::AbstractPipeComponent, p2::OpticalSequence)
    merge(OpticalSequence(), OpticalSequence(p1, p2.optical_components...))
end

function Base.merge(p::OpticalSequence)
    n = length(p.optical_components)
    if n < 2
        p
    elseif n == 2
        merge(p.optical_components...)
    else
        p_merge = merge(OpticalSequence(), p)
        while p_merge != p
            p = p_merge
            p_merge = merge(OpticalSequence(), p)
        end
        p_merge
    end
end

function Base.merge(p1::OpticalSequence, p2::OpticalSequence)
    n1 = length(p1.optical_components)
    n2 = length(p2.optical_components)
    if n2 == 0
        p1
    else
        if n1 > 0
            p_merged = merge(last(p1.optical_components), first(p2.optical_components))
        else
            p_merged = OpticalSequence(first(p2.optical_components))
        end
        head = OpticalSequence(p1.optical_components[1:(n1 - 1)]...,
                               p_merged.optical_components...)
        tail = OpticalSequence(p2.optical_components[2:end]...)
        merge(head, tail)
    end
end

# FourierPhase, FourierMask

simplify(p::Union{FourierPhase, FourierMask}) = OpticalSequence(get_sequence(p)...)

# ASProp, ShiftProp

simplify(p::Union{ASProp, ShiftProp}) = OpticalSequence(get_sequence(p)...)
