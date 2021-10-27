module BigMat

using LinearAlgebra, Base, Random
    import LinearAlgebra: norm

    # functions that are available for the user
    export MPT, size, order, length, rank, norm,
           MPT_SVD, MPT_ALS, eye, rkrp2tn, leaves2roottrunc,
           mps2vec, mpo2mat,
           KathriRao, transpose, roundTT, 
           mpo2mps, mps2mpo

    # package code
    include("MPT.jl")

    include("functions_computeTN/MPT_SVD.jl")
    include("functions_computeTN/TT_SVD.jl")
    include("functions_computeTN/TT_ALS.jl")
    include("functions_computeTN/MPT_ALS.jl")
    include("functions_computeTN/eyeMPO.jl")
    include("functions_computeTN/rkrp2tn.jl")
    include("functions_computeTN/leaves2roottrunc.jl")

    include("functions_tools/BasicAlgebra.jl")
    include("functions_tools/Contractions.jl")
    include("functions_tools/Unfoldings.jl")    
    include("functions_tools/MatrixAlgebra.jl")
    include("functions_tools/roundTT.jl")


end
