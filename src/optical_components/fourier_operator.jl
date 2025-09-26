struct FourierOperator{M, S, P} <: AbstractPureComponent{M}
    p_f::P
    s::S
    direct::Bool

    function FourierOperator(p_f::P, s::S, direct::Bool) where {S, P}
        new{Static, S, P}(p_f, s, direct)
    end

    function FourierOperator(p_f::FFTPlans, direct::Bool)
        s = size(p_f.ft)
        @assert s == size(p_f.ift)
        d = fftdims(p_f.ft)
        @assert d == fftdims(p_f.ift)
        P = typeof(p_f)
        S = Val{(s, d)}
        new{Static, S, P}(p_f, S(), direct)
    end

    function FourierOperator(u::ScalarField{U, Nd}, direct::Bool) where {Nd, U}
        u_plan = similar(u.electric)
        p_f = make_fft_plan(u_plan, Tuple(1:Nd), direct)
        FourierOperator(p_f, direct)
    end
end

get_data(p::FourierOperator) = ()

function propagate!(u::ScalarField, p::FourierOperator, ::Type{<:Direction})
    if p.direct
        compute_ft!(p.p_f, u)
    else
        compute_ift!(p.p_f, u)
    end
end

function propagate(u::ScalarField, p::FourierOperator, ::Type{<:Direction})
    propagate!(copy(u), p, direction)
end

function Base.merge(
        p1::FourierOperator{Static, S},
        p2::FourierOperator{Static, S}
) where {S}
    if p1.direct != p2.direct
        OpticalSequence()
    else
        OpticalSequence(p1, p2)
    end
end
