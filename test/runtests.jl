using Revise
using BigMat
using Test
using Random
using LinearAlgebra
using Optim
using SparseArrays


@testset "BigMat.jl" begin
    include("MPT.jl")
    include("ComputeTN.jl")
    include("MatrixAlgebra.jl") # inversion
end

