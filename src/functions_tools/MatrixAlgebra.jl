    function mldivide(A::MPT{4},b::MPT{3},x0::MPT{3},maxiter::Int64)
    ## DESCRIPTION
    # Inverts a matrix in mpo format, by solving Ax = b, where x is the unknown
    # 
    # INPUT
    # A         Matrix in mpo format that we would like to invert
    # b         Right-hand side in TT-format
    # x0        Intitial guess for x0
    # maxiter   Maximum number of iterations
    # 
    # OUPUT
    # x     Inverse of A in mpo format
    #
    # Sources: Dolgov, Sovastyanov: Alternating minimal energy methods for 
    # linear systems in higher dimensions, 2014.
    #
    # February 2021, Clara Menzen
    ###############################

        order = ndims(b);
        x = x0;
        #if orthog == true # orthoganilize core 2 to N
        #    for n = N:-1:2
        #        x = shiftMPTnorm(x,n,-1);
        #    end
        #end

        for iter = 1:maxiter
            swipe = [collect(1:order)... collect(order-1:2)...];
            #Dir   = [ones(1,order-1) -ones(1,order-1)];
            for k = 1:length(swipe)
                n    = swipe[k];
                #dir  = Dir[k];
                
                An   = createAn(A,x,n);
                bn   = createbn(b,x,n);
                x[n] = reshape(An\bn,size(x[n]));
                
                #if orthog == true
                #    x = shiftMPTnorm(x,n,dir);
                #end
            end
        end
        return x
    
    end

    function createAn(A,x,n)
        order = ndims(A);
        rnksA = rank(A);
        rnksx = rank(x);
        sizeA = size(A);
        
        left  = 1;
        for i = 1:n-1
            left = left*contractcores(contractcores(x[i],A[i]),x[i]);
        end

        right = 1;
        for i = order:-1:n+1
            right = contractcores(contractcores(x[i],A[i]),x[i])*right;
        end

        if n == 1
            An = reshape(A[n],(rnksx[n][1]^2*sizeA[n][1]*sizeA[n][2], rnksA[n][2]));
        else
            left = reshape(left,(rnksx[n-1][2]..., rnksA[n][1]..., rnksb[n-1][2]...));
            left = permutedims(left,[1, 3, 2]);
            left = reshape(left,(rnksx[n-1][2]^2, rnksA[n][1]));
            An = left * reshape(A[n],(rnksA[n][1]..., Int(length(A[n])/rnksA[n][1])));
            An = reshape(An,(rnksx[n][1]^2*sizeA[n][1]*sizeA[n][2], rnksA[n][2]));
        end

        if n < order
            right = reshape(right,(rnksx[n+1][1]..., rnksA[n][2]..., rnksb[n+1][1]...));
            right = permutedims(right,[1,3,2]);
            right = reshape(right,(rnksA[n+1][1], rnksb[n+1][1]^2));
            An    = An *right;
            An    = reshape(An,(rnksx[n][1],rnksx[n][1],sizeA[n][1],sizeA[n][2],rnksx[n+1][1],rnksx[n+1][1]));
            An    = permutedims(An,[1,3,5,2,4,6]);
            An    = reshape(An,(rnksx[n][1]*sizeA[n][1]*rnksx[n+1][1],rnksx[n][1]*sizeA[n][2]*rnksx[n+1][1]));
        else
            An    = reshape(An,(rnksx[n][1],rnksx[n][1],sizeA[n][1],sizeA[n][2]));
            An    = permutedims(An,[1 3 2 4]);
            An    = reshape(An,(rnksx[n][1]*sizeA[n][1],rnksx[n][1]*sizeA[n][2]));
        end

    end

    function createbn(b,x,n)
        order = ndims(b);
        rnksx = rank(x);
        rnksb = rank(b);
        sizeb = size(b);

        left  = 1;
        for i = 1:n-1
            left = left*contractcores(x[i],b[i]);
        end

        right = 1;
        for i = order:-1:n+1
            right = contractcores(x[i],b[i])*right;
        end

        if n == 1
            bn = b[n]
        elseif n > 1
            bn = reshape(left,(rnksx[n-1][2]..., rnksb[n-1][2]...)) * reshape(b[n],(rnksb[n][1]..., sizeb[n][1]*rnksb[n][2]...));
        end

        if n < order
            bn = reshape(bn,(rnksx[n][1])*sizeb[n][1], rnksx[n][2]) * reshape(right,(rnksx[n+1][1], rnksb[n+1][1]))';
        end

        bn = reshape(bn,(rnksx[n][1]*sizeb[n][1]*rnksb[n][2]));
    end
