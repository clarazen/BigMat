using BigMat
using Test
using Random

@testset "TensorGP.jl" begin
    include("MPT.jl")
    include("ComputeTN.jl")
end
