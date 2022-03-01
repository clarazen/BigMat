using Revise
using Pkg
Pkg.activate("C:/Users/cmmenzen/.julia/dev/BigMat")
using BigMat
using Test
using Random
using LinearAlgebra
using Optim


@testset "BigMat.jl" begin
    include("MPT.jl")
    include("ComputeTN.jl")
    include("MatrixAlgebra.jl") # inversion
end

