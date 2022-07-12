# Computation of MPO from given block matrices that together from a sparse matrix containing those blocks in specified locations
function sparseblockTTm(blocks::Vector{Matrix{Float64}},blocklocations::Matrix{Int64},middlesizes::Matrix{Int64},maxrank::Int64,acc::Float64)

    rowsizes = middlesizes[1,:];
    colsizes = middlesizes[2,:];
    cores0 = [zeros(1, rowsizes[e], colsizes[e], 1) for e in 1:length(rowsizes)]; 
    sparseblockmpo = MPT(cores0);
    # compute cores
    for a = 1:length(blocks)
        cores = Vector{Array{Float64,4}}(undef,length(rowsizes));   
        # first core: block matrix
        cores[1] = reshape(blocks[a],(1, rowsizes[1], colsizes[1], 1));
        # second to last core: vectorized E matrix:
        # Zooming in from largest block to smallest
        xloc = blocklocations[a,1]
        yloc = blocklocations[a,2]
        remainingrows = prod(rowsizes[2:end-1])
        remainingcols = prod(colsizes[2:end-1])
        for e = length(rowsizes):-1:2
            xlocindiv              = Int(ceil(xloc/remainingrows)) # x location in division
            ylocindiv              = Int(ceil(yloc/remainingcols)) # y location in division

            E                      = zeros(rowsizes[e],colsizes[e]);
            E[xlocindiv,ylocindiv] = 1;
            cores[e]               = reshape(E,(1, rowsizes[e], colsizes[e], 1));

            xloc                   = xloc-(xlocindiv-1)*remainingrows
            yloc                   = yloc-(ylocindiv-1)*remainingcols
            remainingrows          = remainingrows/rowsizes[e-1]
            remainingcols          = remainingcols/colsizes[e-1]
        end
        
        sparseblockmpo = sparseblockmpo + MPT(cores);
        if maximum(maximum(rank(sparseblockmpo))) > maxrank
            sparseblockmpo = mpo2mps(sparseblockmpo);
            sparseblockmpo = roundTT(sparseblockmpo,acc);
            sparseblockmpo = mps2mpo(sparseblockmpo,middlesizes);
        end
    end
    return sparseblockmpo
end



