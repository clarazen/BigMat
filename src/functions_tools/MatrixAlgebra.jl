import Base: \, transpose  
    
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
            right = contractcores(contractcores(x[i],A[i]),x[i])*right;
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

###############################################################################
    function transpose(mpo::MPO)
        N     = order(mpo);
        cores = Vector{Array{Float64,4}}(undef,N);
        for i = 1: order(mpo)
            cores[i] = permutedims(mpo[i],[1,3,2,4]);
        end
        return MPT(cores)
    end