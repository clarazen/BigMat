import Base: transpose  
import LinearAlgebra: diag

    function transpose(mpo::MPO)
        N     = order(mpo);
        cores = Vector{Array{Float64,4}}(undef,N);
        for i = 1: order(mpo)
            cores[i] = permutedims(mpo[i],[1,3,2,4]);
        end
        return MPT(cores)
    end

    function transpose(tt::MPS)
        N     = order(tt);
        cores = Vector{Array{Float64,4}}(undef,N);
        for i = 1: order(tt)
            cores[i] = reshape(tt[i],(rank(tt)[i][1],1,size(tt)[i][1],rank(tt)[i][2]));
        end
        return MPT(cores)
    end


###################
function diag(ttm::MPT{4})
    # create diagonal TTm from diagonal as TT
    cores  = Vector{Array{Float64,3}}(undef,order(ttm));
    for i = 1:order(ttm)
        cores[i] = zeros(rank(ttm)[i][1],size(ttm)[i][1],rank(ttm)[i][2])
        for ri = 1:rank(ttm)[i][1]
            for rii = 1:rank(ttm)[i][2]
                cores[i][ri,:,rii] = diag(ttm[i][ri,:,:,rii])
            end
        end
    end
    return MPT(cores)
end

