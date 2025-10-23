using FluxOptics
using Zygote
using Test
using LinearAlgebra
using Statistics

@testset "FluxOptics.jl" begin
    include("gridutils_test.jl")
    include("modes_test.jl")
    include("fields_test.jl")
    include("optical_components_test.jl")
    include("optimisers_test.jl")
    include("metrics_test.jl")
end
