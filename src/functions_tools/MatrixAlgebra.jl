import Base: \, transpose  
import LinearAlgebra: diag

    function \(A::MPO,b::Vector)
        middleind = Matrix(reduce(vcat,transpose.(collect.(size(A))))[:,1]');
        btt,err = MPT_SVD(b,middleind,0.0);
        return A\btt
    end
    
    function \(A::MPO,b::MPS)
    ## DESCRIPTION
    # Inverts a matrix in mpo format, by solving Ax = b, where x is the unknown
    # 
    # INPUT
    # A         Matrix in mpo format that we would like to invert
    # b         Right-hand side in TT-format
    # 
    # OUPUT
    # x         solution of linear system Ax = b
    #
    # Sources: Dolgov, Sovastyanov: Alternating minimal energy methods for 
    # linear systems in higher dimensions, 2014.
    #
    # February + August 2021, Clara Menzen
    ######################################

        N     = order(b);
        cores = Vector{Array{Float64,3}}(undef,N);
        maxiter = 10;
        # creating site-N canonical initial tensor train
        for i = 1:N-1 
            tmp = qr(rand(size(b[i],1)*size(b[i],2), size(b[i],3)));
            cores[i] = reshape(Matrix(tmp.Q),size(b[i]));
        end
        cores[N] = reshape(rand(size(b[N],1)*size(b[N],2)),(size(b[N],1), size(b[N],2), 1));
        x = MPT(cores);

        for iter = 1:maxiter
            swipe = [collect(1:N)... collect(N-1:2)...];
            Dir   = Int.([ones(1,N-1) -ones(1,N-1)]);
            for k = 1:length(swipe)
                n    = swipe[k];
                dir  = Dir[k];
                
                An   = createAn(A,x,n);
                bn   = createbn(b,x,n);
                x[n] = reshape(An\bn,size(x[n]));
                
                shiftMPTnorm(x,n,dir);       
            end
        end
        return x
    
    end

    function createAn(A,x,n)
        N     = order(A);
        rnksA = rank(A);
        rnksx = rank(x);
        sizeA = size(A);
        
        # create left side of An
        left  = 1;
        for i = 1:n-1
            left = left*contractcores(contractcores(x[i],A[i]),x[i]);
        end
        if n > 1
            left = reshape(left,(rnksx[n-1][2]..., rnksA[n][1]..., rnksx[n-1][2]...));
            left = permutedims(left,[1, 3, 2]);
            left = reshape(left,(rnksx[n-1][2], rnksx[n-1][2], rnksA[n][1]));
        end

        # create right side of An
        right = 1;
        for i = N:-1:n+1
            right = contractcores(contractcores(x[i],A[i]),x[i])*right
        end
        if n < N
            right = reshape(right,(rnksx[n+1][1]..., rnksA[n][2]..., rnksx[n+1][1]...));
            right = permutedims(right,[2, 1, 3]);
            right = reshape(right,(rnksA[n+1][1], rnksx[n+1][1], rnksx[n+1][1]));
        end

        # compute left*An*right
        if n == 1
            A1 = reshape(A[1],(sizeA[1][1], sizeA[1][2], rnksA[1][2]));
            An = contractmodes(A1,right,[3 1],[1 2; 2 3]);
        elseif n == N
            AN = reshape(A[N],(rnksA[N][1],sizeA[N][1], sizeA[N][2]));
            An = contractmodes(left,AN,[3 1],[1 2; 2 3]);
        else
            An = contractmodes(left,A[n],[3 1],[1 2; 2 3; 0 4]);
            An = contractmodes(An,right,[3 1],[1 2; 2 3]);
        end
    end

    function createbn(b,x,n)
        N     = order(b);
        rnksx = rank(x);
        rnksb = rank(b);
        sizeb = size(b);

        left  = 1;
        for i = 1:n-1
            left = left*contractcores(x[i],b[i]);
        end

        right = 1;
        for i = N:-1:n+1
            right = contractcores(x[i],b[i])*right;
        end

        if n == 1
            bn = b[n]
        elseif n > 1
            bn = reshape(left,(rnksx[n-1][2]..., rnksb[n-1][2]...)) * reshape(b[n],(rnksb[n][1]..., sizeb[n][1]*rnksb[n][2]...));
        end

        if n < N
            bn = reshape(bn,(rnksx[n][1])*sizeb[n][1], rnksx[n][2]) * reshape(right,(rnksx[n+1][1], rnksb[n+1][1]))';
        end

        bn = reshape(bn,(rnksx[n][1]*sizeb[n][1]*rnksb[n][2]));
    end
##############################################################################

function approxpseudoinverse(A::MPT{4},ϵ::Float64,δ::Float64)
    # Algorithm 1 from Lee et al: Regularized computation of approximate pseudoinverse ...

    # Initialize P
    N     = order(A)
    sizes = size(A)
    rnks  = rank(A)
    cores = Vector{Array{Float64,4}}(undef,N)
    for i = 1:N
        tmp      = rand(rnks[i][1], sizes[i][1]*sizes[i][2]*rnks[i][2])
        F        = qr(tmp')
        cores[i] = reshape(Matrix(F.Q)',(rnks[i][1], sizes[i][1], sizes[i][2], rnks[i][2]))
    end
    P     = MPT(cores)
    Ptt   = mpo2mps(P)
    rnksP = rank(P)
    # Compute L1,L2 and R1,R2
    L1    = Vector{Array{Float64,4}}(undef,N)
    L2    = Vector{Array{Float64,2}}(undef,N)
    R1    = Vector{Array{Float64,4}}(undef,N)
    R2    = Vector{Array{Float64,2}}(undef,N)
    L1[1] = reshape([1],(1, 1, 1, 1))
    L2[1] = reshape([1],(1, 1))
    R1[N] = reshape([1],(1, 1, 1, 1))
    R2[N] = reshape([1],(1, 1))
    r1    = 1
    r2    = 1
    for n = N-1:-1:2
        Z1,Z2 = getZ(P[n+1],A[n+1])
        r1 = Z1*r1
        R1[n] = reshape(r1,(rnksP[n][1],rnks[n][1],rnks[n][1],rnksP[n][1]))
        r2 = Z2*r2
        R2[n] = reshape(r2,(rnksP[n][1],rnks[n][1]))
    end
    #while sth < ϵ
    for i = 1:100
        l1t = 1; l2t = 1;
        for n = 1:N-2
            # solve local optimization problem
            Ā = getĀ(L1[n],A[n],A[n+1],R1[n+1])
            b̄ = getb̄(L2[n],A[n],A[n+1],R2[n+1])
            f(p) = p'*Ā*p - 2*p'*b̄ + 0.1*p'*p
            res =  optimize(f,(Ptt[n]*Ptt[n+1])[:],NGMRES())
            p2 = Optim.minimizer(res)
            P2 = reshape(p2,(rnksP[n][1]*sizes[n][1]*sizes[n][2], sizes[n+1][1]*sizes[n+1][2]*rnksP[n+2][1]))
            # Matrix factorization by svd 
            F     = svd!(P2) 
            r     = length(F.S)
            sv2   = cumsum(reverse(F.S).^2)
            tr    = Int(findfirst(sv2 .> δ^2))-1
            if tr > 0
                r     = length(F.S) - tr 
            end
            U = F.U[:,1:r]
            S = Diagonal(F.S[1:r])
            V = F.V[:,1:r]
            rnksP[n][1] = minimum([r,rnksP[n][1]])
            P[n]   = reshape(U,(rnks[n][1],sizes[n][1],sizes[n][1],rnks[n][2]))
            P[n+1] = reshape(S*V',(rnks[n+1][1],sizes[n+1][1],sizes[n+1][1],rnks[n+1][2]))

            Z1,Z2   = getZ(P[n],A[n])
            l1t     = l1t*Z1
            l2t     = l2t*Z2
            L1[n+1] = reshape(l1t,(rnksP[n+1][1],rnks[n+1][1],rnks[n+1][1],rnksP[n+1][1]))
            L2[n+1] = reshape(l2t,(rnksP[n+1][1],rnks[n+1][1]))
        end
        r1  = 1; r2  = 1;
        for n = N-1:-1:2
            # solve local optimization problem
            Ā    = getĀ(L1[n],A[n],A[n+1],R1[n+1])
            b̄    = getb̄(L2[n],A[n],A[n+1],R2[n+1])
            f(p) = p'*Ā*p - 2*p'*b̄ + 0.1*p'*p
            res  =  optimize(f,(Ptt[n]*Ptt[n+1])[:],NGMRES())
            p2   = Optim.minimizer(res)
            P2   = reshape(p2,(rnksP[n][1]*sizes[n][1]*sizes[n][2], sizes[n+1][1]*sizes[n+1][2]*rnksP[n+1][2]))
            # Matrix factorization by svd 
            F     = svd!(Matrix(P2')) 
            r     = length(F.S)
            sv2   = cumsum(reverse(F.S).^2)
            tr    = Int(findfirst(sv2 .> δ^2))-1
            if tr > 0
                r     = length(F.S) - tr 
            end
            U = F.U[:,1:r]
            S = Diagonal(F.S[1:r])
            V = F.V[:,1:r]
            rnksP[n][1] = minimum([r,rnksP[n][1]])
            P[n+1]   = reshape(U',(rnks[n+1][1],sizes[n+1][1],sizes[n+1][1],rnks[n+1][2]))
            P[n] = reshape(V*S,(rnks[n][1],sizes[n][1],sizes[n][1],rnks[n][2]))

            Z1,Z2 = getZ(P[n+1],A[n+1])
            r1    = Z1*r1
            r2    = Z2*r2
            R1[n] = reshape(r1,(rnksP[n][1],rnks[n][1],rnks[n][1],rnksP[n][1]))
            R2[n] = reshape(r2,(rnksP[n][1],rnks[n][1]))
        end
    end
    return P
end

function getZ(P::Array,A::Array)
    Z1 = zeros((size(P,1)*size(A,1))^2,(size(P,4)*size(A,4))^2)
    for i = 1:size(A,2)
        for j = 1:size(A,3)
            for ii = 1:size(A,2)
                for jj = 1:size(A,3)
                    tmp = kron(P[:,i,j,:],A[:,i,jj,:])
                    tmp = kron(tmp,A[:,ii,jj,:])
                    tmp = kron(tmp,P[:,ii,j,:])
                    Z1 = Z1 + tmp
                end
            end
        end
    end
    Z2 = zeros(size(P,1)*size(A,1),size(P,4)*size(A,4))
    for i = 1:size(A,2)
        for j = 1:size(A,3)
            tmp = kron(P[:,i,j,:],A[:,i,j,:])
            Z2 = Z2 + tmp
        end
    end
    return Z1,Z2
end

function getĀ(L::Array{Float64,4},A4::Array{Float64,4},B4::Array{Float64,4},R::Array{Float64,4})
    A   = reshape(A4,(size(A4,1),size(A4,2)*size(A4,3),size(A4,4)))
    B   = reshape(B4,(size(B4,1),size(B4,2)*size(B4,3),size(B4,4)))
    tmp = contractmodes(L,A,[2 1])
    tmp = permutedims(tmp,[1 4 5 2 3])
    tmp = contractmodes(tmp,A,[4 1])
    tmp = permutedims(tmp,[1 2 3 5 6 4])
    tmp = contractmodes(tmp,B,[3 1])
    tmp = permutedims(tmp,[1 2 6 7 3 4 5])
    tmp = contractmodes(tmp,B,[6 1])
    tmp = permutedims(tmp,[1 2 3 4 5 7 8 6])
    tmp = contractmodes(tmp,R,[4 2; 7 3])
    tmp = permutedims(tmp,[1 2 3 7 6 4 5 8])
    sz  = size(L,1)*size(A,2)*size(B,2)*size(R,1)
    Ā   = reshape(tmp,(sz,sz))
    return Ā
end

function getb̄(L::Matrix,A::Array{Float64,4},B::Array{Float64,4},R::Matrix)
    tmp = contractmodes(L,A,[2 1])
    tmp = contractmodes(tmp,B,[4 1])
    tmp = contractmodes(tmp,R,[6 2])
    b̄   = tmp[:]
    return b̄
end

###############################################################################
    function transpose(mpo::MPO)
        N     = order(mpo);
        cores = Vector{Array{Float64,4}}(undef,N);
        for i = 1: order(mpo)
            cores[i] = permutedims(mpo[i],[1,3,2,4]);
        end
        return MPT(cores)
    end

    function transpose(tt::MPS)
        N     = order(tt);
        cores = Vector{Array{Float64,4}}(undef,N);
        for i = 1: order(tt)
            cores[i] = reshape(tt[i],(rank(tt)[i][1],1,size(tt)[i][1],rank(tt)[i][2]));
        end
        return MPT(cores)
    end


###################
function diag(ttm::MPT{4})
    # create diagonal TTm from diagonal as TT
    cores  = Vector{Array{Float64,3}}(undef,order(ttm));
    for i = 1:order(ttm)
        cores[i] = zeros(rank(ttm)[i][1],size(ttm)[i][1],rank(ttm)[i][2])
        for ri = 1:rank(ttm)[i][1]
            for rii = 1:rank(ttm)[i][2]
                cores[i][ri,:,rii] = diag(ttm[i][ri,:,:,rii])
            end
        end
    end
    return MPT(cores)
end

