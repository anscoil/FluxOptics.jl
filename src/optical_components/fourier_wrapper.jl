function FourierWrapper(p_f::FFTPlans, wrapped_components::Vararg{AbstractPipeComponent})
    ft = FourierOperator(p_f, true)
    ift = FourierOperator(p_f, false)
    OpticalSequence(ft, wrapped_components..., ift)
end

function FourierWrapper(u::ScalarField{U, Nd},
                        wrapped_components::Vararg{AbstractPipeComponent}) where {Nd, U}
    u_plan = similar(u.electric)
    p_f = make_fft_plans(u_plan, Tuple(1:Nd))
    FourierWrapper(p_f, wrapped_components...)
end

include("fourier_phase.jl")
include("fourier_mask.jl")
