function MPT_SVD(arrayaux::Array{Float64},middlesizes::Matrix{Int64},acc::Float64)
    tensor = Array{Float64};
    array  = copy(arrayaux);
    if size(middlesizes,1) == 1
        tensor = reshape(array,Tuple(middlesizes));
        return TT_SVD(tensor,acc);
    else
        sizes   = Tuple(reshape(middlesizes',(length(middlesizes),1)));
        array   = reshape(array,sizes);
        permind = Tuple([ (i-1)*size(middlesizes,2)+j for j in 1:size(middlesizes,2) for i in 1:size(middlesizes,1) ]);
        array   = permutedims(array,permind);
        resind  = Tuple([prod(col) for col in eachcol(middlesizes)]);
        tensor  = reshape(array,resind)
        mps,err = TT_SVD(tensor,acc);
        rnks    = rank(mps);
        return MPT( [reshape(mps[i],(rnks[i][1], middlesizes[:,i]..., rnks[i][2])) for i = 1:order(mps)] ), err
    end   
end

# TT-ALS for diagonal Matrix
function MPT_SVD(diagonal::Diagonal,middlesizes::Matrix,acc::Float64)
    # initial ttm with diagonal core 
    N      = length(middlesizes);
    tensor = reshape(diag(diagonal),Tuple(middlesizes));
    tt,err = TT_SVD(tensor,acc);
    cores  = Vector{Array{Float64,4}}(undef,N);
    rnks   = rank(tt);
    for i = 1:N
        cores[i] = zeros(rnks[i][1],middlesizes[i],middlesizes[i],rnks[i][2])
        for ri = 1:rnks[i][1]
            for rii = 1:rnks[i][2]
                cores[i][ri,:,:,rii] = Matrix(Diagonal(tt[i][ri,:,rii]))
            end
        end
    end
    return MPT(cores)
end

# TT-ALS for symmetric Matrix
function MPT_SVD(sym::Symmetric,middlesizes::Matrix,acc::Float64)
    # initial ttm with diagonal core 
    d          = Int(length(middlesizes)/2)
    symM       = Matrix(sym)
    tensor     = reshape(symM,Tuple(middlesizes));
    Uttm,S,err = TT_SVD(tensor,d,acc);
    r          = rank(Uttm)

    cores   = Vector{Array{Float64,3}}(undef,2*d)
    coresL  = Vector{Array{Float64,4}}(undef,d)
    coresU  = Vector{Array{Float64,4}}(undef,d)

    di     = collect(d:-1:1)
    for i = 1:d
        cores[i]   = Uttm[i]
        cores[d+i] = permutedims(Uttm[di[i]],[3,2,1])

        if i<d
            coresL[i]  = reshape(Uttm[i],(size(Uttm[i],1),size(Uttm[i],2),1,size(Uttm[i],3)))
            coresU[i]  = coresL[i]
        end
    end

    coresU[d]  = reshape(cores[d],(size(cores[d],1),size(cores[d],2),size(cores[d],3),1))
    cores[d]   = nmodeproduct(Matrix(sqrt.(Diagonal(S))),cores[d],3)
    cores[d+1] = nmodeproduct(Matrix(sqrt.(Diagonal(S))),cores[d+1],1)
    coresL[d]  = reshape(cores[d],(size(cores[d],1),size(cores[d],2),size(cores[d],3),1))

    return MPT(cores),MPT(coresL),MPT(coresU),S,err
end
