@testset "TensorGP.jl" begin
    # tests for MPT_SVD, MPT_SVD, contract
    # vector to MPS
    v1      = [1.0; 4.5; 2; 7; 3; 7; 3.3; 10; 6; 8; 1.1; 4.6; 1; 6; 2; 7; 12; 4.5; 1; 0; 0; 3; 6; 2.1];
    v2      = [1.0; 4.5; 2; 7; 3; 7; 3.3; 10; 6; 8; 1.1; 4.6; 1; 6; 2; 7; 12; 4.5; 1; 0; 0; 3; 6; 2.1];
    tt1     = MPT_SVD(v1,[2 3 4],0.0);
    tt2,res = MPT_ALS(v2,[2 3 4],[1,2,4,1]);
    @test (norm(tt1) - norm(tt2)) < 1e-13
    @test norm(mps2vec(tt1) - v1)/norm(v1) < 1e-13

    # matrix to MPO 
    M1 = v1*v1';
    M2 = v2*v2';
    mpo1 = MPT_SVD(M1,[2 3 4; 2 3 4],0.0);
    mpo2,res = MPT_ALS(M2,[2 3 4; 2 3 4],[1,4,16,1]);
    @test (norm(mpo1) - norm(mpo2)) < 1e-14

    @run mpo2mat(mpo1)
    @test norm(mpo2mat(mpo1) - M1)/norm(M1) < 1e-14
    @test norm(mpo1)-norm(mpo2mat(mpo1)) < 1e-14

    # check if error of lr approx is really lower than the error bound set in eps
    # let TT-SVD also return the error that is made by throwing away the singualar values
    tt3 = MPT_SVD(v1,[2 3 4],0.1);
    @test norm(mps2vec(tt3)-v1)/norm(v1) <= 0.1
end