module BigMat

using LinearAlgebra, Base, Random, SparseArrays #, MATLAB
    import LinearAlgebra: norm

    # functions that are available for the user
    export MPT, size, order, length, rank, norm,
           getelement, getrow, getcol, outerprod,
           MPT_SVD, TTm_ALS, TT_ALS, eye, khr2ttm, sparseblockTTm,
           shiftMPTnorm,
           mps2vec, mpo2mat, kr2mat,
           KhatriRao, transpose, roundTT, 
           krtimesttm,
           nmodeproduct, contractmodes,
           mpo2mps, mps2mpo,
           unfold, diag, matrixbyvector, vectorbymatrix

    # package code
    include("MPT.jl")

    include("functions_computeTN/MPT_SVD.jl")
    include("functions_computeTN/TT_SVD.jl")
    include("functions_computeTN/TT_ALS.jl")
    include("functions_computeTN/TTm_ALS.jl")
    include("functions_computeTN/eyeTTm.jl")
    include("functions_computeTN/khr2ttm.jl")
    include("functions_computeTN/sparseblockTTm.jl")

    include("functions_tools/BasicAlgebra.jl")
    include("functions_tools/Contractions.jl")
    include("functions_tools/Unfoldings.jl")    
    include("functions_tools/MatrixAlgebra.jl")
    include("functions_tools/roundTT.jl")


end
