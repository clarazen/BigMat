module BigMat

using LinearAlgebra, Base, Random
    import LinearAlgebra: norm

    # functions that are available for the user
    export MPT, size, order, length, rank, norm,
           MPT_SVD, MPT_ALS, eye, rkrp2tn,
           mps2vec, mpo2mat

    # package code
    include("MPT.jl")

    include("functions_computeTN/MPT_SVD.jl")
    include("functions_computeTN/TT_SVD.jl")
    include("functions_computeTN/TT_ALS.jl")
    include("functions_computeTN/MPT_ALS.jl")
    include("functions_computeTN/eyeMPO.jl")
    include("functions_computeTN/rkrp2tn.jl")

    include("functions_tools/BasicAlgebra.jl")
    include("functions_tools/Contractions.jl")
    include("functions_tools/Unfoldings.jl")    
    include("functions_tools/MatrixAlgebra.jl")


end
