function MPT_ALS(array::Array{Float64},middlesizes::Matrix{Int64},rnks::Vector{Int64})
    tensor = Array{Float64};
    if size(middlesizes,1) == 1
        tensor = reshape(array,Tuple(middlesizes));
        mpt    = TT_ALS(tensor,rnks);
    else
        sizes   = Tuple(reshape(middlesizes',(length(middlesizes),1)));
        array   = reshape(array,sizes);
        permind = Tuple([ (i-1)*size(middlesizes,2)+j for j in 1:size(middlesizes,2) for i in 1:size(middlesizes,1) ]);
        array   = permutedims(array,permind);
        resind  = Tuple([prod(col) for col in eachcol(middlesizes)]);
        tensor  = reshape(array,resind)
        mps     = TT_ALS(tensor,rnks);
        rnks    = rank(mps);
        mpt = MPT( [reshape(mps[i],(rnks[i][1], middlesizes[:,i]..., rnks[i][2])) for i = 1:order(mps)] );
    end   
    return mpt
end

# TT-ALS for diagonal Matrix
function MPT_ALS(diagonal::Diagonal,middlesizes::Matrix,rnks::Vector)
    # initial ttm with diagonal core 
    N      = length(rnks)-1;
    tensor = reshape(diag(diagonal),Tuple(middlesizes));
    tt     = TT_ALS(tensor,rnks);
    cores  = Vector{Array{Float64,4}}(undef,N);
    for i = 1:N
        cores[i] = zeros(rnks[i],middlesizes[i],middlesizes[i],rnks[i+1])
        for ri = 1:rnks[i]
            for rii = 1:rnks[i+1]
                cores[i][ri,:,:,rii] = Matrix(Diagonal(tt[i][ri,:,rii]))
            end
        end
    end
    return MPT(cores)
end

function MPT_ALS(array::Array{Float64},middlesizes::Matrix{Int64},rnks::Vector{Int64},mpt0::MPT)
    tensor = Array{Float64};
    if size(middlesizes,1) == 1
        tensor = reshape(array,Tuple(middlesizes));
        mpt    = TT_ALS(tensor,rnks,mpt0);
    else
        sizes   = Tuple(reshape(middlesizes',(length(middlesizes),1)));
        array   = reshape(array,sizes);
        permind = Tuple([ (i-1)*size(middlesizes,2)+j for j in 1:size(middlesizes,2) for i in 1:size(middlesizes,1) ]);
        array   = permutedims(array,permind);
        resind  = Tuple([prod(col) for col in eachcol(middlesizes)]);
        tensor  = reshape(array,resind);
        tt0     = mpo2mps(mpt0);
        mps     = TT_ALS(tensor,rnks,tt0);
        rnks    = rank(mps);
        mpt = MPT( [reshape(mps[i],(rnks[i][1], middlesizes[:,i]..., rnks[i][2])) for i = 1:order(mps)] );
    end   
    return mpt
end
