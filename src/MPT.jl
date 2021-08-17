
    # MPT type
    struct MPT{N}
        cores::Vector{Array{Float64,N}}
        function MPT(cores::Vector{Array{Float64,N}}) where N
            new{ndims(cores[1])}(cores) 
        end
    end

    # create aliases for MPS and MPO
    const MPS = MPT{3};
    const MPO = MPT{4};

    # middle sizes of each core
    function Base.size(mpt::MPT)
        sizes = [size(core) for core in mpt.cores];
        [sizes[i][2:end-1] for i in 1:length(sizes)]
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

