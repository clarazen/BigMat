function MPT_SVD(arrayaux::Array{Float64},middlesizes::Matrix{Int64},acc::Float64)
    tensor = Array{Float64};
    array  = copy(arrayaux);
    if size(middlesizes,1) == 1
        tensor = reshape(array,Tuple(middlesizes));
        TT_SVD(tensor,acc);
    else
        sizes   = Tuple(reshape(middlesizes',(length(middlesizes),1)));
        array   = reshape(array,sizes);
        permind = Tuple([ (i-1)*size(middlesizes,2)+j for j in 1:size(middlesizes,2) for i in 1:size(middlesizes,1) ]);
        array   = permutedims(array,permind);
        resind  = Tuple([prod(col) for col in eachcol(middlesizes)]);
        tensor  = reshape(array,resind)
        mps     = TT_SVD(tensor,acc);
        rnks    = rank(mps);
        MPT( [reshape(mps[i],(rnks[i][1], middlesizes[:,i]..., rnks[i][2])) for i = 1:order(mps)] );
    end   
end
