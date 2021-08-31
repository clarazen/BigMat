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
        tensor  = reshape(array,resind)
        mps     = TT_ALS(tensor,rnks,mpt0);
        rnks    = rank(mps);
        mpt = MPT( [reshape(mps[i],(rnks[i][1], middlesizes[:,i]..., rnks[i][2])) for i = 1:order(mps)] );
    end   
    return mpt
end
