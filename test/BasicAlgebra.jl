# test outerprod between 2 vectors
a = rand(16);
tta,err = MPT_SVD(a,[2 2 2 2],0.0);
ttb     = mpo2mps(eye([2 2; 2 2]));
b       = mps2vec(ttb);
ttb,err = MPT_SVD(b,[2 2 2 2],0.0);
norm(outerprod(tta,ttb)-a*b')


# outerprod between 2 matrices
A        = rand(8,8);
ttma,err = MPT_SVD(A,[2 2 2;2 2 2],0.0);
ttmb     = eye([3 3 3; 3 3 3]);

C    = kron(Matrix(I,27,27),A);
tmp  = reshape(C,(2,2,2,3,3,3,2,2,2,3,3,3));
tmp  = permutedims(tmp,[1,4,2,5,3,6,7,10,8,11,9,12]);
test = reshape(tmp,216,216);

norm(mpo2mat(outerprod(ttma,ttmb))-test)
