@testset "BigMat.jl" begin
    # tests for \ (solving of linear system Ax = b and A*inv(K) = I)
    
    A = rand(64,64);
    A = A*A'/2;
    b = rand(64,1);
        
    mps,err = MPT_SVD(b,[4 4 4],0.0);
    mpo,err = MPT_SVD(A,[4 4 4; 4 4 4],0.0);
    @test norm(mps2vec(mpo\mps)-A\b)/norm(A\b) < 1e-5

    # solve directly for inverse with A*inv(A) = i
    id        = eye([4 4 4; 4 4 4]);
    mpo1      = MPT([mpo[1:3]..., id[1:3]...]); 
    id        = Matrix(I,64,64);
    id        = 1.0*id[:];
    idvec,err = MPT_SVD(id,[4 4 4 4 4 4],0.0);
    invA      = sum\idvec;
    reshape(mps2vec(invA),(64,64));
    @test norm(lyap-inv(A))/norm(inv(A)) < 1e-5

    # solve Lyaponov equation ((I ⊗ A)+(A^T ⊗ I))vec(X) = 2vec(I)
    id = eye([4 4 4; 4 4 4]);
    mpo1 = MPT([mpo[1:3]..., id[1:3]...]);
    mpoT = transpose(mpo);
    mpo2 = MPT([id[1:3]..., mpoT[1:3]...]);

    sum = mpo1 + mpo2;
    id = Matrix(I,64,64);
    id = 2.0*id[:];
    idvec,err = MPT_SVD(id,[4 4 4 4 4 4],0.0);

    lyap = sum\idvec;
    lyap = reshape(mps2vec(lyap),(64,64));
    @test norm(lyap-inv(A))/norm(inv(A)) < 1e-5
    
end