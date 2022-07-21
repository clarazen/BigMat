module BigMat

using LinearAlgebra, Base, Random, SparseArrays,MATLAB
    import LinearAlgebra: norm

    # functions that are available for the user
    export MPT, size, order, length, rank, norm,
           getelement, getrow, getcol, outerprod,
           MPT_SVD, TTm_ALS, TT_ALS, eye, rkrp2tn,
           sparseblockTTm, ALS_krtt, BayALS_krtt,
           kr2mat,
           mps2vec, mpo2mat,
           KhatriRao, transpose, roundTT, 
           krtimesttm,
           nmodeproduct, contractmodes,
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
    include("functions_computeTN/sparseblockTTm.jl")
    include("functions_computeTN/ALS_krtt.jl")
    include("functions_computeTN/BayALS_krtt.jl")

    include("functions_tools/BasicAlgebra.jl")
    include("functions_tools/Contractions.jl")
    include("functions_tools/Unfoldings.jl")    
    include("functions_tools/MatrixAlgebra.jl")
    include("functions_tools/roundTT.jl")
    include("functions_tools/TNrSVD.jl")


end
