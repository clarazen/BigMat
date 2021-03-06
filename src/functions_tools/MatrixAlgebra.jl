import Base: \, transpose  
import LinearAlgebra: diag

    function \(A::MPO,b::Vector)
        middleind = Matrix(reduce(vcat,transpose.(collect.(size(A))))[:,1]');
        btt,err   = MPT_SVD(b,middleind,0.0);
        return A\btt
    end

    function \(A::MPO,B::MPO,eps::Float64)
        # uses AMEn from TT-toolbox
        sizeA1 = size(A,true)[1,:];
        sizeB2 = size(B,true)[2,:];
        Ittm   = eye([sizeB2'; sizeB2'])
        A_     = outerprod(A,Ittm)
        b_     = mpo2mps(B)
        
        return mps2mpo(\(A_,b_,eps),[sizeA1';sizeB2'])
    end

    function \(A::MPO,b::MPS,eps::Float64)
        # uses AMEn from TT-toolbox
        D    = Float64(order(A))
        cores_A = Float64[]
        cores_b = Float64[]
        for d = 1:order(A)
            append!(cores_A,A[d][:])
            append!(cores_b,b[d][:])
        end
        rnksA   = rank(A,true)
        rnksb   = rank(b,true)
        sizeA1  = size(A,true)[1,:]
        sizeA2  = size(A,true)[2,:]
        sizeA   = Float64.(sizeA1 .* sizeA2)
        sizeb   = Float64.(size(b,true))
        scA     = cumsum([1, length(A)[1]...])
        scb     = cumsum([1, length(b)[1]...])
        # make A in tt-toolbox format
        mat"Att      = tt_tensor();"
        mat"Att.d    = $D;"
        mat"Att.r    = $rnksA;"
        mat"Att.n    = $sizeA;"
        mat"Att.core = $cores_A;"
        mat"Att.ps   = $scA;"
        mat"Attm     = tt_matrix(Att,$sizeA1,$sizeA2)"
        # make b in tt-toolbox format
        mat"btt      = tt_tensor();"
        mat"btt.d    = $D;"
        mat"btt.r    = $rnksb;"
        mat"btt.n    = $sizeb;"
        mat"btt.core = $cores_b;"
        mat"btt.ps   = $scb;"
        
        mat"sol_tt  = amen_solve2(Attm, btt, $eps);"
        
        mat"vecsol  = sol_tt.core"
        mat"rnkssol = sol_tt.r"
        mat"sz1sol  = sol_tt.n"
        mat"scsol   = sol_tt.ps"

        vecsol = @mget vecsol
        r      = Int.(@mget rnkssol)
        sz1    = Int.(@mget sz1sol)
        sc     = Int.(@mget scsol)

        D = Int.(D)
        cores = Vector{Array{Float64,3}}(undef,D)
        for i = 1:D
            cores[i] = reshape(vecsol[sc[i]:sc[i+1]-1],r[i],sz1[i],r[i+1])
        end
        
        return MPT(cores)
    end

    function \(A::MPO,B::MPO,rnks::Vector)
        Ittm = eye([size(B,true)[2,:]'; size(B,true)[2,:]'])
        A_ = outerprod(A,Ittm)
        b  = mpo2mps(B)
        tmp  = \(A_,b,rnks)
        return mps2mpo(tmp,[size(A,true)[1,:]';size(B,true)[2,:]'])
    end

    function \(A::MPO,b::MPS,rnks::Vector)
        N     = order(b);
        cores = Vector{Array{Float64,3}}(undef,N);
        maxiter = 10;
        # creating site-N canonical initial tensor train
        for i = 1:N-1 
            tmp = qr(rand(rnks[i]*size(b[i],2), rnks[i+1]));
            cores[i] = reshape(Matrix(tmp.Q),(rnks[i],size(b[i],2),rnks[i+1]));
        end
        cores[N] = reshape(rand(rnks[N]*size(b[N],2)),(rnks[N], size(b[N],2), 1));
        x = MPT(cores);

        for iter = 1:maxiter
            if N == 2
                swipe = [1, 2];
            else
                swipe = [collect(1:N)... collect(N-1:2)...];
            end
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
            if N == 2
                swipe = [1, 2];
            else
                swipe = [collect(1:N)... collect(N-1:2)...];
            end
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
           bn = reshape(bn,(rnksx[n][1])*sizeb[n][1], rnksb[n][2]) * reshape(right,(rnksx[n+1][1], rnksb[n+1][1]))';
        end

        return bn[:]
    end
##############################################################################




function approxpseudoinverse(A::MPT{4},??::Float64,??::Float64)
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
    #while sth < ??
    for i = 1:100
        l1t = 1; l2t = 1;
        for n = 1:N-2
            # solve local optimization problem
            A?? = getA??(L1[n],A[n],A[n+1],R1[n+1])
            b?? = getb??(L2[n],A[n],A[n+1],R2[n+1])
            f(p) = p'*A??*p - 2*p'*b?? + 0.1*p'*p
            res =  optimize(f,(Ptt[n]*Ptt[n+1])[:],NGMRES())
            p2 = Optim.minimizer(res)
            P2 = reshape(p2,(rnksP[n][1]*sizes[n][1]*sizes[n][2], sizes[n+1][1]*sizes[n+1][2]*rnksP[n+2][1]))
            # Matrix factorization by svd 
            F     = svd!(P2) 
            r     = length(F.S)
            sv2   = cumsum(reverse(F.S).^2)
            tr    = Int(findfirst(sv2 .> ??^2))-1
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
            A??    = getA??(L1[n],A[n],A[n+1],R1[n+1])
            b??    = getb??(L2[n],A[n],A[n+1],R2[n+1])
            f(p) = p'*A??*p - 2*p'*b?? + 0.1*p'*p
            res  =  optimize(f,(Ptt[n]*Ptt[n+1])[:],NGMRES())
            p2   = Optim.minimizer(res)
            P2   = reshape(p2,(rnksP[n][1]*sizes[n][1]*sizes[n][2], sizes[n+1][1]*sizes[n+1][2]*rnksP[n+1][2]))
            # Matrix factorization by svd 
            F     = svd!(Matrix(P2')) 
            r     = length(F.S)
            sv2   = cumsum(reverse(F.S).^2)
            tr    = Int(findfirst(sv2 .> ??^2))-1
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

function getA??(L::Array{Float64,4},A4::Array{Float64,4},B4::Array{Float64,4},R::Array{Float64,4})
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
    A??   = reshape(tmp,(sz,sz))
    return A??
end

function getb??(L::Matrix,A::Array{Float64,4},B::Array{Float64,4},R::Matrix)
    tmp = contractmodes(L,A,[2 1])
    tmp = contractmodes(tmp,B,[4 1])
    tmp = contractmodes(tmp,R,[6 2])
    b??   = tmp[:]
    return b??
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

function krtimesttm(kr::Vector{Matrix},ttm::MPT{4},??::Float64)
    # computes the produt of two matrices, where the first one (kr) has a 
    # row-wise Khatri-Rao structure and the second (ttm) is a TTm. 
    # The trucation parameter ?? is for rounding step

    D        = order(ttm)
    N        = size(kr[1],1)
    cores    = Vector{Array{Float64,4}}(undef,D)
    cores[1] = nmodeproduct(kr[1],ttm[1],3)
    for d = 2:D 
        Md2  = size(ttm[d-1],3) 
        Md1  = size(ttm[d-1],2) 
        Mdd2 = size(ttm[d],3)
        Mdd1 = size(ttm[d],2)
        Rd   = size(cores[d-1],1)
        Rdd  = size(cores[d-1],4)
        Rddd = size(ttm[d],4)
        
        tmp = KhatriRao(unfold(cores[d-1],[3],"right"),Matrix(kr[d]'),2)
        Tmp = reshape(tmp,(Mdd2,Rd,Md1,Rdd,N))
        Tmp = permutedims(Tmp,[2,3,5,4,1])
        Tmp = contractmodes(Tmp,ttm[d],[4 1; 5 3])
        tmp = reshape(Tmp,(Rd*Md1,N*Mdd1*Rddd))

        # rounding the rank
        F    = svd!(tmp); 
        R    = length(F.S);
        err2 = 0;
        sv2  = cumsum(reverse(F.S).^2);
        tr   = Int(findfirst(sv2 .> ??^2))-1;
        if tr > 0
            R = length(F.S) - tr;
            err2 += sv2[tr];
        end

        cores[d-1] = reshape(F.U[:,1:R],(Rd,Md1,1,R))
        cores[d]   = permutedims(reshape(Diagonal(F.S[1:R])*F.Vt[1:R,:],(R,N,Mdd1,Rddd)),[1,3,2,4])
    end
    return MPT(cores)
end

function krtimesttm(kr::Vector{Matrix},ttm::MPT{4})
    # computes the produt of two matrices, where the first one (kr) has a 
    # row-wise Khatri-Rao structure and the second (ttm) is a TTm. 

    D  = order(ttm)
    N  = size(kr[1],1)
    
    Tmp     = nmodeproduct(kr[1],ttm[1][1,:,:,:],2)
    tmp     = KhatriRao(unfold(Tmp,[2],"right"),Matrix(kr[2]'),2)
    newdims = (size(ttm[2],3),size(ttm[1],2),size(ttm[1],4),N)
    Tmp     = reshape(tmp,newdims)
    Tmp     = contractmodes(Tmp,ttm[2],[1 3; 3 1])

    for d = 2:D-1     
        tmp     = KhatriRao(unfold(Tmp,[d],"right"),Matrix(kr[d+1]'),2)
        newdims = (size(ttm[d+1],3),newdims[2:d]...,size(ttm[d],2),size(ttm[d],4),N)
        Tmp     = reshape(tmp,newdims)
        Tmp     = contractmodes(Tmp,ttm[d+1],[1 3; d+2 1])  
    end
    return unfold(Tmp,[D],"right")
end

# only for testing, do not use in practice
function krtimesrank1(W::Vector{Matrix},K::Vector{Matrix})
    D      = size(W,1)
    N      = size(W[1],1)
    M      = size(K[1],1)

    coresW = Vector{Array{Float64,4}}(undef,D)
    coresK = Vector{Array{Float64,4}}(undef,D)

    coresW[1] = reshape(W[1]',(1,1,M,N))
    for d = 2:D
        coresW[d] = zeros(N,1,M,N)
        for m = 1:M
            coresW[d][:,1,m,:] = Diagonal(W[d][:,m])
        end
    end

    coresW[D] = permutedims(coresW[D],[1,4,3,2])
    for d = 1:D
        coresK[d] = reshape(K[d],(1,M,M,1))
    end

    return MPT(coresW)*MPT(coresK)
end