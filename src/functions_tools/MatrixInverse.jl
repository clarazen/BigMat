import Base: \, transpose  
import LinearAlgebra: diag

    function \(A::MPT{4},b::Vector)
        middleind = Matrix(reduce(vcat,transpose.(collect.(size(A))))[:,1]');
        btt,err   = MPT_SVD(b,middleind,0.0);
        return A\btt
    end

    function \(A::MPT{4},B::MPT{4},eps::Float64)
        sizeA1 = size(A,true)[1,:];
        sizeB2 = size(B,true)[2,:];
        Ittm   = eye([sizeB2'; sizeB2'])
        A_     = outerprod(A,Ittm)
        b_     = mpo2mps(B)
        
        return mps2mpo(\(A_,b_,eps),[sizeA1';sizeB2'])
    end
#=
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
=#
    function \(A::MPT{4},B::MPT{4},rnks::Vector)
        Ittm = eye([size(B,true)[2,:]'; size(B,true)[2,:]'])
        A_   = outerprod(A,Ittm) + transpose(outerprod(A,Ittm))
        b    = mpo2mps(B)
        tmp  = \(A_,b,rnks)
        return mps2mpo(tmp,[size(A,true)[1,:]';size(B,true)[2,:]'])
    end

    function \(A::MPT{4},b::MPT{3},rnks::Vector)
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
    
    function \(A::MPT{4},b::MPT{3})
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