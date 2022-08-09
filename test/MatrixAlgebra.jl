#@testset "BigMat.jl" begin
    using LinearAlgebra
    # tests for \ (solving of linear system Ax = b and A*inv(K) = I)
    
    A = rand(64,64);
    A = A*A'/2;
    b = rand(64,1);
    c = rand(64,1);
    mps,err = MPT_SVD(b,[4 4 4],0.0);
    mps_2,err = MPT_SVD(c,[4 4 4],0.0);

    mpo,err = MPT_SVD(A,[4 4 4; 4 4 4],0.0)
    mpo\mps
    @test norm(mps2vec(mpo\mps)-A\b)/norm(A\b) < 1e-5

    # solve directly for inverse with A*inv(A) = I
    id        = eye([4 4 4; 4 4 4]);
    mpo1      = MPT([mpo[1:3]..., id[1:3]...]); 
    id        = Matrix(I,64,64);
    id        = 1.0*id[:];
    idvec,err = MPT_SVD(id,[4 4 4 4 4 4],0.0);
    invA      = mpo1\idvec;
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

    # test of column-wise and row-wise Kronecker product (KathriRao)
    A = rand(3,4);
    B = rand(3,4);
    C = rand(3,4);
    @test size(KathriRao(A,B,1),2) == 16
    @test size(KathriRao(A,B,2),1) == 9
    
    @test norm(A.*B.*C) < norm(KathriRao(A,KathriRao(B,C,2),2))

    # test matrix multiplication
    @test norm(mps2vec(mpo*mpo*mps) - A*A*b)/norm(A*A*b) < 1e-10

    A = randn(8,8)
    B = randn(8,8)
    Attm,err = MPT_SVD(A,[2 2 2;2 2 2],0.0)
    AB = *(Attm,B,3)
    norm(AB-A*B)/norm(A*B)
    ABA = AB*transpose(Attm)
    norm(ABA-A*B*A')/norm(A*B*A')

    # test pseudo inverse (I think does not work yet)
    @run approxpseudoinverse(Attm,0.0,0.0)
    P = approxpseudoinverse(Attm,0.0,0.0)
    norm(inv(A) - mpo2mat(P))/norm(inv(A))

    # test vectorbymatrix
    A = randn(8,2)
    Attm,err = MPT_SVD(A,[2 2 2;1 1 2],0.0)
    b = randn(8)
    btt,err = MPT_SVD(b,[2 2 2],0.0)
    norm(vectorbymatrix(Attm,btt) - b'*A)/norm(b'*A)

    # test matrixbyvector
    matrixbyvector(Attm,btt)
    norm(matrixbyvector(Attm,btt) - A'*b)/norm(A'*b)  

       


#end


# test matrix inverse where result is matrix
coresK = Vector{Array{Float64,4}}(undef,3)
for i = 1:3
    K = rand(2,2); K = K*K'
    coresK[i] = reshape(K,(1,2,2,1))
end
Kttm = MPT(coresK);
Ittm = eye([2 2 2;2 2 2]);

S        = randn(8,8); S = S*S';
Sttm,err = MPT_SVD(S,[2 2 2; 2 2 2],1e-15);

#amen
ref = mpo2mat(Sttm)\mpo2mat(Kttm)
sol1 =  mpo2mat(\(Sttm,Kttm,0.0000001))
norm(sol1-ref)/norm(ref)

# \ with given ranks
sol2 = mpo2mat(\(Sttm,Kttm,[1,4,4,1]))
norm(sol2-ref)/norm(ref)
##############


# test with amen
using MATLAB
mat"addpath('C:/Users/cmmenzen/surfdrive/Code/Code from others/TT-Toolbox')"
mat"addpath('C:/Users/cmmenzen/surfdrive/Code/Code from others/TT-Toolbox/solve')"
mat"addpath('C:/Users/cmmenzen/surfdrive/Code/Code from others/TT-Toolbox/core')"
A = randn(8,8); 
B = randn(8,8); #B = B*B';
b = randn(8,1); 

Attm,err = MPT_SVD(A,[2 2 2 ; 2 2 2],0.0)
btt,err  = MPT_SVD(b,[2 2 2; 1 1 1],0.0)
Btt,err  = MPT_SVD(B,[2 2 2; 2 2 2],0.0)

test = norm(mpo2mat(\(Attm,Btt,1e-10)))
ref  = norm(mpo2mat(Attm)\mpo2mat(Btt))


test = norm(mpo2mat(\(Attm,btt,1e-7)))
ref  = norm(mpo2mat(Attm)\mpo2mat(btt))
norm(test-ref)/norm(ref)


