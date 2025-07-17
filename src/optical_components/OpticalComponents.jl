module OpticalComponents

using ..Types

export make_fft_plans
export propagate!, backpropagate!

function propagate!(args...)
    error("Not implemented");
end
function backpropagate!(args...)
    error("Not implemented");
end

include("freespace.jl")
export ASProp, RSProp

include("phasemask.jl")
export Phase

end
