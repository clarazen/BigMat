
    # MPT type
    mutable struct MPT{N}
        cores::Vector{Array{Float64,N}}
        normcore::Int64
        function MPT(cores::Vector{Array{Float64,N}},normcore::Int64) where N
            new{ndims(cores[1])}(cores,normcore) 
        end
    end
    MPT(cores) = MPT(cores,0);

    # aliases for MPS and MPO
    const MPS = MPT{3};
    const MPO = MPT{4};

    # middle sizes of each core
    function Base.size(mpt::MPT)
        sizes = [size(core) for core in mpt.cores];
        [sizes[i][2:end-1] for i in 1:length(sizes)]
    end

    function Base.size(tt::MPT{3},s::Bool)
        [size(core)[2] for core in tt.cores];
    end

    function Base.size(ttm::MPT{4},s::Bool)
        middlesizes = zeros(2,order(ttm))
        middlesizes[1,:] = [size(core)[2] for core in ttm.cores]
        middlesizes[2,:] = [size(core)[3] for core in ttm.cores]
        Int.(middlesizes)
    end

    # number of cores 
    function order(mpt::MPT)
        collect(size(mpt.cores))[1]
    end

    # number of elements in each core and overall
    function Base.length(mpt::MPT)
        lengths = [length(core) for core in mpt.cores];
        [lengths,[sum(lengths)]]
    end

    # ranks of each core
    function LinearAlgebra.rank(mpt::MPT)
        sizes = [size(core) for core in mpt.cores];
        [[sizes[i][1], sizes[i][end]] for i in 1:length(sizes)]    
    end

    function LinearAlgebra.rank(mpt::MPT,s::Bool)
        sizes = [size(core) for core in mpt.cores];
        [[sizes[i][1] for i in 1:length(sizes)]..., 1]    
    end

    function LinearAlgebra.norm(mps::MPS)
        N   = order(mps);
        mptnorm = contractcores(mps[1],mps[1]);
        for i = 2:N
            mptnorm = mptnorm*contractcores(mps[i],mps[i]);
        end
        return sqrt(mptnorm[1])
    end

    function LinearAlgebra.norm(mpo::MPO)
        mps = mpo2mps(mpo);
        norm(mps)
    end

    # indexing to get and set a core in an MPT
    Base.IndexStyle(::Type{<:MPT}) = IndexLinear() 
    Base.getindex(mpt::MPT, i::Int) = mpt.cores[i] # tt[1] gives the first core
    Base.getindex(mpt::MPT, range::UnitRange{Int64}) = [mpt.cores[i] for i in range]
    Base.setindex!(mpt::MPT,v,i::Int) = setindex!(mpt.cores, v, i) # tt[1] = rand(1,5,3) sets the first core


    # indexing to get an element or row/column of the reconstructed vector or matrix

    function getelement(tt::MPT{3},row::Int)
        sz = size(tt);
        d  = ();
        el = 1;
        for i = 1:length(sz)
            d = (d...,1:sz[i][1])
        end
        linear = LinearIndices(d);
        index = findall(x->x==row,linear)[1];
        for i = 1:length(sz)
            el = el*tt[i][:,index[i],:];
        end
        return el[1]
    end

    function getelement(ttm::MPT{4},row::Int,col::Int)
        sz = size(ttm);
        d1  = ();
        d2  = ();
        el = 1;
        for i = 1:length(sz)
            d1 = (d1...,1:sz[i][1])
            d2 = (d2...,1:sz[i][2])
        end
        linear1 = LinearIndices(d1);
        index1 = findall(x->x==row,linear1)[1];
        linear2 = LinearIndices(d2);
        index2 = findall(x->x==col,linear2)[1];
        for i = 1:length(sz)
            el = el*ttm[i][:,index1[i],index2[i],:];
        end
        return el[1]
    end


    function getcol(ttm::MPT{4},col::Int)
        sz = size(ttm);
        d  = ();
        for i = 1:length(sz)
            d = (d...,1:sz[i][1])
        end
        linear = LinearIndices(d);
        index = findall(x->x==col,linear)[1];

        cores = Vector{Array{Float64,3}}(undef,order(ttm));
        for i = 1:length(sz)
            cores[i] = ttm[i][:,:,index[i],:];
        end
        return MPT(cores)
    end

    function getrow(ttm::MPT{4},row::Int)
        sz = size(ttm);
        d  = ();
        for i = 1:length(sz)
            d = (d...,1:sz[i][2])
        end
        linear = LinearIndices(d);
        index = findall(x->x==row,linear)[1];

        cores = Vector{Array{Float64,3}}(undef,order(ttm));
        for i = 1:length(sz)
            cores[i] = ttm[i][:,index[i],:,:];
        end
        return MPT(cores)
    end