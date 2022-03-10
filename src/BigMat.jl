module BigMat

using LinearAlgebra, Base, Random, Optim
    import LinearAlgebra: norm

    # functions that are available for the user
    export MPT, size, order, length, rank, norm,
           getelement, getrow, getcol, outerprod,
           MPT_SVD, TTm_ALS, TT_ALS, eye, rkrp2tn,
           mps2vec, mpo2mat,
           KathriRao, transpose, roundTT, 
           mpo2mps, mps2mpo,
           unfold, diag, matrixbyvector, vectorbymatrix,
           TNrSVD,
           approxpseudoinverse

    # package code
    include("MPT.jl")

    include("functions_computeTN/MPT_SVD.jl")
    include("functions_computeTN/TT_SVD.jl")
    include("functions_computeTN/TT_ALS.jl")
    include("functions_computeTN/TTm_ALS.jl")
    include("functions_computeTN/eyeMPO.jl")
    include("functions_computeTN/rkrp2tn.jl")

    include("functions_tools/BasicAlgebra.jl")
    include("functions_tools/Contractions.jl")
    include("functions_tools/Unfoldings.jl")    
    include("functions_tools/MatrixAlgebra.jl")
    include("functions_tools/roundTT.jl")
    include("functions_tools/TNrSVD.jl")


end
