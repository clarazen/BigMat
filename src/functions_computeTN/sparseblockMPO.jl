# Computation of MPO from given block matrices that together from a sparse matrix containing those blocks in specified locations
function sparseblockMPO(blocks::Vector{Matrix{Float64}},blocklocations::Matrix{Int64},middleindices::Matrix{Int64},maxrank::Int64,acc::Float64)

    rowsizes = middleindices[1,:];
    colsizes = middleindices[2,:];
    cores0 = [zeros(1, rowsizes[e]*colsizes[e], 1) for e in 1:length(rowsizes)]; 
    sparseblockmpo = MPT(cores0);
    # compute cores
    for a = 1:length(blocks)
        cores = Vector{Array{Float64,3}}(undef,length(rowsizes));   
        # first core: block matrix
        cores[1] = reshape(blocks[a],(1, rowsizes[1]*colsizes[1], 1));
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
            cores[e]               = reshape(E,(1, rowsizes[e]*colsizes[e], 1));

            xloc                   = xloc-(xlocindiv-1)*remainingrows
            yloc                   = yloc-(ylocindiv-1)*remainingcols
            remainingrows          = remainingrows/rowsizes[e-1]
            remainingcols          = remainingcols/colsizes[e-1]
        end
        
        sparseblockmpo = sparseblockmpo + MPT(cores);
        #if max(ranks(sparseblockmpo)) > maxrank
        #    sparseblockmpo = roundTT(sparseblockmpo,acc);
        #end
    end
    return sparseblockmpo
end

## test ##
blocks = [[1 2; 3 4],[7 7; 6 1],[1 0;5 5.0],[5 3; 7 4]];
blocklocations = [1 1; 2 2; 3 3; 4 4];
middleindices = [2 2 2; 2 2 2];
maxrank = 1000;
acc = 1e-10;

Kmpo = sparseblockMPO(blocks,blocklocations,middleindices,maxrank,acc)
K    = approxTensor(Kmpo)
K    = reshape(K,(2,2,2,2,2,2));
K    = permutedims(K,(1,3,5,2,4,6));
K    = reshape(K,(8,8));

