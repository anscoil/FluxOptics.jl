module OpticalComponents

using ..Types

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
