    # todo check if number of cores are the same, check if summing index matches
    
    import Base: +, -
    import Base: *
    # summation of two MPTs
    function +(mpt1::MPT,mpt2::MPT)
        N        = order(mpt1);
        cores    = Vector{Array{Float64,ndims(mpt1[1])}}(undef,N);
        cores[1] = cat(mpt1[1], mpt2[1],dims=ndims(mpt1[1]));
        cores[N] = cat(mpt1[N], mpt2[N],dims=1);
        sizes = size(mpt1);
        rnks1 = rank(mpt1);
        rnks2 = rank(mpt2);
        for i = 2:N-1    
            row1     = cat(mpt1[i], zeros((rnks1[i][1],sizes[i]...,rnks2[i][2])),dims=ndims(mpt1[i]));
            row2     = cat(zeros((rnks2[i][1],sizes[i]...,rnks1[i][2])), mpt2[i],dims=ndims(mpt1[i]));
            cores[i] = cat(row1,row2,dims=1);
        end
                
        MPT(cores);
    end

    function +(mpo::MPO,mat::Matrix{Float64})
        return mpo2mat(mpo) + mat
    end

    function +(mat::Matrix{Float64},mpo::MPO)
        return +(mpo,mat)
    end

    function -(mpo1::MPO,mpo2::MPO)
        return +(mpo1,-1.0*mpo2)
    end

    function -(mpo::MPO,mat::Matrix{Float64})
        return mpo2mat(mpo) - mat
    end

    function -(mat::Matrix{Float64},mpo::MPO)
        return mat - mpo2mat(mpo)
    end

    # multiplying MPT by a number
    function *(mpt::MPT,a::Float64)
        # if mpt in site-k-canonical format that multiply norm core with scalar
        mpt[1] = a*mpt[1];
        return mpt
        #MPT(pushfirst!(mpt[2:ndims(mpt)],a*mpt[1]))
    end

    function *(a::Float64,mpt::MPT)
        *(mpt,a)
    end

    # matrix by vector product (returns mps)
    function *(mpo::MPT{4},mps::MPT{3})
        N = order(mpo);
        cores = Vector{Array{Float64,3}}(undef,N);
        for i = 1:N
            cores[i] = contractcores(mpo[i],mps[i]);
        end
        MPT(cores)
    end

    # matrix by matrix product (returns mpo)
    function *(mpo1::MPT{4},mpo2::MPT{4})
        N = order(mpo1);
        cores = Vector{Array{Float64,4}}(undef,N);
        for i = 1:N
            cores[i] = contractcores(mpo1[i],mpo2[i]);
        end
        MPT(cores)
    end

    function *(mpo1::MPT{4},mpo2::MPT{4},ϵ)
        N = order(mpo1);
        cores = Vector{Array{Float64,4}}(undef,N);
        for i = 1:N
            cores[i] = contractcores(mpo1[i],mpo2[i]);
        end
        MPT(cores)
    end

    # vector by matrix product !! check if it zip line
    function *(mps::MPT{3},mpo::MPT{4})
        N = order(mpo);
        cores = Vector{Array{Float64,3}}(undef,N);
        for i = 1:N
            cores[i] = contractcores(mps[i],mpo[i]);
        end
        MPT(cores)
    end

    # vector by vector product (returns scalar)
    function *(mps1::MPT{3},mps2::MPT{3})
        N = order(mps1);
        c = 1;
        for i = 1:N
            c = c*contractcores(mps1[i],mps2[i]);
        end
        return c[1]
    end

    # mpo times vector
    function *(mpo::MPT{4},a::Vector)
        D      = order(mpo)
        mps    = mpo2mps(mpo)
        tmp    = nmodeproduct(a,mpo[D],3)
        mps[D] = reshape(tmp,(size(tmp,1),size(tmp,2),size(tmp,3)))

        return mps
    end

    function outerprod(tt1::MPT{3},tt2::MPT{3})
        N     = order(tt1);
        cores = Vector{Array{Float64,4}}(undef,N);
        r1 = rank(tt1);
        r2 = rank(tt2);
        s1 = size(tt1);
        s2 = size(tt2);
        for i = 1:N
            g1  = reshape(tt1[i],(length(tt1[i]), 1));
            g2  = reshape(tt2[i],(length(tt2[i]), 1));
            tmp = kron(g2,g1);
            tmp = reshape(tmp,(r1[i][1], s1[i][1], r1[i][2], r2[i][1], s2[i][1], r2[i][2]));
            tmp = permutedims(tmp,[1 4 2 5 3 6]);
            tmp = reshape(tmp,(r1[i][1]*r2[i][1], s1[i][1], s2[i][1], r1[i][2]*r2[i][2]));
            cores[i]  = tmp;
        end
        return MPT(cores);
    end

    function outerprod(ttm1::MPT{4},ttm2::MPT{4})
        tt1     = mpo2mps(ttm1)
        tt2     = mpo2mps(ttm2)
        outertt = outerprod(tt1,tt2)
        D       = order(ttm1)
        cores   = Vector{Array{Float64,4}}(undef,D)
        sz1     = size(ttm1,true)
        sz2     = size(ttm2,true)
        rk1     = rank(ttm1,true)
        rk2     = rank(ttm2,true)
        for d = 1:D
            newsize  = (rk1[d]*rk2[d],sz1[1,d],sz1[2,d],sz2[1,d],sz2[2,d],rk1[d+1]*rk2[d+1])
            tmp      = reshape(outertt[d],newsize)
            tmp      = permutedims(tmp,[1 2 4 3 5 6])
            newsize  = (rk1[d]*rk2[d],sz1[1,d]*sz2[1,d],sz1[2,d]*sz2[2,d],rk1[d+1]*rk2[d+1])
            cores[d] = reshape(tmp,newsize)
        end
        return MPT(cores)
    end

    function *(core1::Array{Float64,3},core2::Array{Float64,3})
        size1 = size(core1);
        size2 = size(core2);
        reshape(unfold(core1,[3],"right")*unfold(core2,[1],"left"),(size1[1],size1[2]*size2[2], size2[3]))
    end

    function *(core1::Array{Float64,4},core2::Array{Float64,4})
        s1 = size(core1);
        s2 = size(core2);
        reshape(unfold(core1,[4],"right")*unfold(core2,[1],"left"),(s1[1],s1[2]*s1[3]*s2[2]*s2[3], s2[4]))
    end

    function *(cores::Vector{Array{Float64,3}})
        tmp = cores[1]*cores[2]
        for i = 3:length(cores)
            tmp = tmp*cores[i]
        end
        return tmp[1,:,:]
    end

    function *(ttm::MPT{4},A::Matrix,ind::Int) # for now work only if second index is in last core
        d = order(ttm)
        cores = Vector{Array{Float64,4}}(undef,d)
        for i = 1:d-1
            cores[i] = ttm[i]
        end
        #ttm[ind] = nmodeproduct(A,ttm[ind],3)
        tmp = ttm[ind]
        tmp = reshape(tmp,(rank(ttm)[ind][1]*size(ttm)[ind][1],size(ttm)[ind][2]))
        tmp = tmp*A
        cores[ind] = reshape(tmp,(rank(ttm)[ind][1],size(ttm)[ind][1],size(A,2),1))
        return MPT(cores)
    end

    function krtimesttm(kr::Vector{Matrix},ttm::MPT{4},ϵ::Float64)
        # computes the produt of two matrices, where the first one (kr) has a 
        # row-wise Khatri-Rao structure and the second (ttm) is a TTm. 
        # The trucation parameter ϵ is for rounding step
    
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
            tr   = Int(findfirst(sv2 .> ϵ^2))-1;
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
        Tmp     = contractmodes(Tmp,ttm[2],[1 3; 3 1]) #this is slow
    
        for d = 2:D-1     
            tmp     = KhatriRao(unfold(Tmp,[d],"right"),Matrix(kr[d+1]'),2)
            newdims = (size(ttm[d+1],3),newdims[2:d]...,size(ttm[d],2),size(ttm[d],4),N)
            Tmp     = reshape(tmp,newdims)
            Tmp     = contractmodes(Tmp,ttm[d+1],[1 3; d+2 1])  #this is slow
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

    function KhatriRao(A::Matrix{Float64},B::Matrix{Float64},dims::Int64)
        if dims == 1
            C = zeros(size(A,1),size(A,2)*size(B,2));
            @inbounds @simd for i = 1:size(A,1)
                @views kron!(C[i,:],A[i,:],B[i,:])
            end
        elseif dims == 2
            C = zeros(size(A,1)*size(B,1),size(A,2));
            @inbounds @simd for i = 1:size(A,2)
                @views kron!(C[:,i],A[:,i],B[:,i])
            end
        end

        return C
    end

    function KhatriRao(A::Matrix{Float64},B::SparseMatrixCSC,dims::Int64)
        if dims == 1
            C = zeros(size(A,1),size(A,2)*size(B,2));
            @inbounds @simd for i = 1:size(A,1)
                @views kron!(C[i,:],A[i,:],B[i,:])
            end
        elseif dims == 2
            C = zeros(size(A,1)*size(B,1),size(A,2));
            @inbounds @simd for i = 1:size(A,2)
                @views kron!(C[:,i],A[:,i],B[:,i])
            end
        end

        return C
    end

    function KhatriRao(A::SparseMatrixCSC,B::Matrix{Float64},dims::Int64)
        if dims == 1
            C = zeros(size(A,1),size(A,2)*size(B,2));
            @inbounds @simd for i = 1:size(A,1)
                @views kron!(C[i,:],A[i,:],B[i,:])
            end
        elseif dims == 2
            C = zeros(size(A,1)*size(B,1),size(A,2));
            @inbounds @simd for i = 1:size(A,2)
                @views kron!(C[:,i],A[:,i],B[:,i])
            end
        end

        return C
    end