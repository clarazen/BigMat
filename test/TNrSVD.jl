#@testset "BigMat.jl" begin
    A        = randn(20,5)
    Attm,err = MPT_SVD(A,[5 2 2; 5 1 1],0.0)

    # test qr
    @run Qttm,R = qr(Attm,0.0)
    Q = mpo2mat(Qttm)
    norm(qr(A).R-R)
    norm(Array(qr(A).Q)-mpo2mat(Qttm))

    Q = randTTmsubspaceiter(q,A,O,acc)
    W,S,V = svd(Attm)
    
    @run U,S,V    = TNrSVD(Attm,30,10,10,0.0)

#end