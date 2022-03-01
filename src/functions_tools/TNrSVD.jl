import LinearAlgebra: svd,qr

function TNrSVD(A::MPT{4},R::Int,s::Int,q::Int,acc::Float64)
# Algorihtm 5.5 from Batselier 2018
#######################################

    # generate random Matrix (Lemma 5.1)
    O = genrandTTm(size(A),R+s)
    # compute orthogonal basis (Alg. 5.4)
    Q = randTTmsubspaceiter(q,A,O,acc)
    # Compute B (subsection 5.3)
    B = transpose(Q)*A
    # Compute econ. SVD (Alg. 5.3)
    W,S,V = svd(B)
    # Compute U
    U = Q
    U[1] = modenprod(U[1],W')

    return U,S,V
end


function genrandTTm(rowsizes::Vector{Tuple{Int64, Int64}},colsize::Int)
    # generate random Matrix (Lemma 5.1)
    d     = length(rowsizes)
    cores = Vector{Array{Float64,4}}(undef,d)
    cores[1] = randn(1,rowsizes[1][2],colsize,1)
    for i = 2:d
        cores[i] = randn(1,rowsizes[i][2],1,1)
    end
    return MPT(cores)
end

function randTTmsubspaceiter(q::Int,A::MPT{4},O::MPT{4},acc::Float64)
    # compute orthogonal basis (Alg. 5.4 + 5.2)
    Y = A*O
    Q = qr(Y)
    for i = 1:q 
        Y = transpose(A)*Q
        Q = qr(Y)
        Q = roundTT(Q,acc)
        Y = A*Q
        Q = qr(Y)
        Q = roundTT(Q,acc)
    end
end

function qr(A::MPT{4},ϵ::Float64)
    d = order(A)
    r = rank(A)
    s = size(A)
    frobnorm    = norm(A); 
    err2        = 0;
    Av = mpo2mps(A)
    
    δ = ϵ / sqrt(d-1) * frobnorm;
    for k = 1:d-1
        F     = svd(unfold(Av[k],[3],"right"),full=false);
        rcurr = length(F.S);
        sv2   = cumsum(reverse(F.S).^2);
        tr    = Int(findfirst(sv2 .> δ^2))-1;
        if tr > 0
            rcurr = length(F.S) - tr;
            err2 += sv2[tr];
        end
        Utr = F.U[:,1:rcurr];
        Str = F.S[1:rcurr];
        Vtr = F.Vt[1:rcurr,:];

        Av[k]   = reshape( Utr,( Int(length(Utr)/(size(Av[k],2)*size(Vtr,1))), size(Av[k],2), size(Vtr,1) ) );
        Av[k+1] = nmodeproduct(Matrix((Vtr'*Diagonal(Str))'),Av[k+1],1);     
    end    

    s1 = [s[i][1] for i = 1:d]
    s2 = [s[i][2] for i = 1:d]
    sz = [s1';s2']
    A = mps2mpo(Av,sz)

    cores = Vector{Array{Float64,4}}(undef,d)
    for i = d:-1:2
        Ai        = reshape(A[i],(r[i][1],s[i][1]*r[i][2]))
        F         = qr(Ai')
        cores[i]  = reshape(Array(F.Q'),(r[i][1],s[i][1],1,r[i][2]))
        A[i-1]    = nmodeproduct(Array(F.R),A[i-1],4)
    end
    A1t = permutedims(A[1],[3 1 2 4])
    A1  = reshape(A1t,(s[1][2], s[1][1]*r[1][2]))
    F   = qr(A1')
    Q1t = reshape(Array(F.Q)',(1, s[1][1], r[1][2], s[1][2]))
    cores[1] = permutedims(Q1t,[1,2,4,3])
    
    return MPT(cores),F.R
end

function svd(A::MPT{4})
    # Algorithm 5.3
    d = order(A)
    r = rank(A)
    s = size(A)
    cores = Vector{Array{Float64,4}}(undef,d)
    for i = d:-1:2
        Ai       = reshape(A[i],(r[i][1],s[i][1]*r[i][2]))
        F        = qr(Ai)
        cores[i] = reshape(Array(F.Q),(r[i][1],s[i][1],1,r[i][2]))
        A[i-1]   = nmodeproduct(Array(F.R),A[i-1],4)
    end   
    A1t      = permutedims(A[1],[3 1 2 4])
    A1       = reshape(A1t,(s[1][2], s[1][1]*r[1][2]))
    W,S,Q1   = svd(A1)
    Q1t      = reshape(Q1,(s[1][2], 1, s[1][1], r[1][2]))
    cores[1] = permutedims(Q1t,[2 3 1 4])

    return W,S,MPT(cores)
end