#@testset "BigMat.jl" begin
    # tests for MPT_SVD, MPT_SVD, contract
    # vector to MPS
    v1      = [1.0; 4.5; 2; 7; 3; 7; 3.3; 10; 6; 8; 1.1; 4.6; 1; 6; 2; 7; 12; 4.5; 1; 0; 0; 3; 6; 2.1];
    v2      = [1.0; 4.5; 2; 7; 3; 7; 3.3; 10; 6; 8; 1.1; 4.6; 1; 6; 2; 7; 12; 4.5; 1; 0; 0; 3; 6; 2.1];
    M1      = randn(64,64)
    M2      = Symmetric(M1*M1')
    
    tt1,err = MPT_SVD(v1,[2 3 4],0.07);
    tt2     = TT_ALS(v2,[2 3 4],[1,2,4,1]);
    ttm1    = TTm_ALS(M1,[4 4 4; 4 4 4],[1,16,16,1]);
    ttm2    = TTm_ALS(M2,[4 4 4; 4 4 4],[1,16,16,1])
    ttm2    = TTm_ALS(Symmetric(M1*M1'),[4 4 4; 4 4 4],[1,3,3,1])

    @test (norm(tt1) - norm(tt2)) < 1e-13
    @test norm(mps2vec(tt1) - v1)/norm(v1) < 1e-13
    @test norm(mpo2mat(ttm1) - M1)/norm(M1) < 1e-13
    @test norm(mpo2mat(ttm2) - mpo2mat(ttm2)') < 1e-10

    # matrix to MPO 
    M1 = v1*v1';
    M2 = v2*v2';
    mpo1,err = MPT_SVD(M1,[2 3 4; 2 3 4],0.0);
    mpo2 = MPT_ALS(M2,[2 3 4; 2 3 4],[1,4,16,1]);
    @test (norm(mpo1) - norm(mpo2)) < 1e-14

    @run mpo2mat(mpo1)
    @test norm(mpo2mat(mpo1) - M1)/norm(M1) < 1e-14
    @test norm(mpo1)-norm(mpo2mat(mpo1)) < 1e-14

    tt3 = MPT_SVD(v1,[2 3 4],0.07);
    @test norm(mps2vec(tt3)-v1)/norm(v1) <= 0.07

    # TT-ALS without orthogonalization needs to be checked. Wihtout pinv, the error is very high
    # maxiter + residulas need to be implemented

    # test of rktp2tn
    #Ũ = [rand(50,40),rand(50,40),rand(50,40)];
    Ũ = [Float64.(Matrix(I,4,4)),Float64.(Matrix(I,4,4)),Float64.(Matrix(I,4,4))];
    ϵ = 0.001;
    Umpo,err2 = rkrp2tn(Ũ,ϵ);
    Uapprox   = mpo2mat(Umpo)
    U12       = zeros(size(Ũ[1],1)^2,size(Ũ[1],2));
    @inbounds @simd for j = 1:size(Ũ[1],2)
        @views kron!(U12[:,j],Ũ[1][:,j],Ũ[2][:,j])
    end
    Utrue = zeros(size(Ũ[1],1)^3,size(Ũ[1],2));
    @inbounds @simd for j = 1:size(Ũ[1],2)
        @views kron!(Utrue[:,j],U12[:,j],Ũ[3][:,j])
    end
    @test norm(Utrue-Uapprox)/norm(Utrue) < ϵ

    # for 2 matrices with sparse matrices 
    Ũ = [sparse(Int.(Matrix(I,4,4))),sparse(Int.(Matrix(I,4,4)))];
    ϵ = 0.0;
    @run rkrp2tn(Ũ,ϵ)
    Umpo,err2 = rkrp2tn(Ũ,ϵ);
    Uapprox   = mpo2mat(Umpo)
    Utrue     = zeros(size(Ũ[1],1)^2,size(Ũ[1],2));
    @inbounds @simd for j = 1:size(Ũ[1],2)
        @views kron!(Utrue[:,j],Ũ[1][:,j],Ũ[2][:,j])
    end
    @test norm(Utrue-Uapprox)/norm(Utrue) <= ϵ


    # get matrix entry from mpo2mat
    Ktest = zeros(24,24);
    linear = LinearIndices((1:2, 1:3, 1:4, 1:2, 1:3, 1:4));
    for i1 = 1:2
        for i2 = 1:3
            for i3 = 1:4
                for j1 = 1:2
                    for j2 = 1:3
                        for j3 = 1:4
                            tmp = mpo2[1][:,i1,j1,:]*mpo2[2][:,i2,j2,:]*mpo2[3][:,i3,j3,:];
                            Ktest[linear[i1,i2,i3,j1,j2,j3]] = tmp[1];
                        end
                    end
                end
            end
        end
    end
    @test norm(Ktest-M2)/norm(M2) < 1e-10

    # TT-SVD for symmetrical matrix
    X, y, f, K  = gensynthdata(1024,1,0.01,1.0,1.0)
    Kttm,Lttm,Uttm,S,err = MPT_SVD(Symmetric(K),[4 4 4 4 4; 4 4 4 4 4],0.1)
    k = mps2vec(Kttm)
    Kt = reshape(k,(4,4,4,4,4,4,4,4,4,4))
    Kt = permutedims(Kt,[1 2 3 4 5 10 9 8 7 6])
    Kapprox = reshape(Kt,(1024, 1024))
    norm(K[:]-Kt[:])/norm(K[:])
    rank(Kttm)

    L  = (Kttm[1]*Kttm[2]*Kttm[3]*Kttm[4]*Kttm[5])[1,:,:]
    L2 = mpo2mat(Lttm)
    norm(L*L' - K)/norm(K)
    norm(L2*L2' - K)/norm(K)
    U = mpo2mat(Uttm)
    norm(U*Diagonal(S)*U'-K)/norm(K)

    # test block TTm
    blocks = [[1 2; 3 4],[7 7; 6 1],[1 2; 5 5.0],[5 3; 7 4]];
    blocklocations = [1 2; 2 2; 3 4; 4 1];
    middleindices = [2 2 2; 2 2 2];
    maxrank = 5;
    acc = 1e-10;
    Kmpo = sparseblockTTm(blocks,blocklocations,middleindices,maxrank,acc)
    K    = sparse(mpo2mat(Kmpo));
    print(K)

    # test Hierarchical Tucker function (does not work yet)
    tensor = rand(5,5,5);
    ranks  = [2,2,2];
    @run leaves2roottrunc(tensor,ranks)

#end