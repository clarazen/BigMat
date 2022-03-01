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
            tmp = kron(g1,g2);
            tmp = reshape(tmp,(r1[i][1], s1[i][1], r1[i][2], r2[i][1], s2[i][1], r2[i][2]));
            tmp = permutedims(tmp,[1 4 2 5 3 6]);
            tmp = reshape(tmp,(r1[i][1]*r2[i][1], s1[i][1], s2[i][1], r1[i][2]*r2[i][2]));
            cores[i]  = tmp;
        end
        return MPT(cores);
    end

    function *(core1::Array{Float64,3},core2::Array{Float64,3})
        size1 = size(core1);
        size2 = size(core2);
        reshape(unfold(core1,[3],"right")*unfold(core2,[1],"left"),(size1[1],size1[2]*size2[2], size2[3]))
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

    function KathriRao(A::Matrix{Float64},B::Matrix{Float64},dims::Int64)
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

