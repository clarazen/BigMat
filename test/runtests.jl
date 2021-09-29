using Revise
# ] activate .
using BigMat
using Test
using Random

# here are all the tests


@testset "BigMat.jl" begin
    include("MPT.jl")
    include("ComputeTN.jl")
    include("MatrixAlgebra.jl") # inversion
end
