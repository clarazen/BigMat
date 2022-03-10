    # matricization of a tensor with indices ind on the lr indicating left or right
    function unfold(tensor::Array{Float64},ind::Vector{Int64},lr::String)
        order    = ndims(tensor);
        newsizes = collect(size(tensor));
        if size([1,2])[1]>1
        deleteat!(newsizes,sort(ind));
        else
            deleteat!(newsizes,ind);
        end
        newperm  = collect(1:order);
        deleteat!(newperm,sort(ind))
        if lr == "left"
            newperm = vcat(ind, newperm);
            tensor  = permutedims(tensor, newperm);
            size_r  = prod(newsizes);
            size_l  = Int(length(tensor)/size_r);
            matrix  = reshape(tensor, (size_l, size_r));    
        elseif lr == "right"
            newperm = vcat(newperm, ind);
            tensor  = permutedims(tensor, newperm);
            size_l  = prod(newsizes);
            size_r  = Int(length(tensor)/size_l);
            matrix  = reshape(tensor, (size_l, size_r));
        end
    end

    # combining middlesizes from mpo to get an mps
    function mpo2mps(mpo::MPT{4})
        cores = Vector{Array{Float64,3}}(undef,order(mpo));
        rnks  = rank(mpo);
        sizes = size(mpo);
        for i = 1:order(mpo)
            cores[i] = reshape(mpo[i],(rnks[i][1], sizes[i][1]*sizes[i][2], rnks[i][2]));
        end
        MPT(cores,mpo.normcore);
    end

    # dividing middlesizes from mps to get an mpo
    function mps2mpo(mps::MPS,middlesizes::Matrix{Int64})
        cores = Vector{Array{Float64,4}}(undef,order(mps));
        rnks  = rank(mps);
        for i = 1:order(mps)
            cores[i] = reshape(mps[i],(rnks[i][1], middlesizes[1,i], middlesizes[2,i], rnks[i][2]));
        end
        MPT(cores,mps.normcore)
    end

    function mps2mpo(mps::MPS,middlesizes::Vector{Tuple{Int64, Int64}})
        cores = Vector{Array{Float64,4}}(undef,order(mps));
        rnks  = rank(mps);
        for i = 1:order(mps)
            cores[i] = reshape(mps[i],(rnks[i][1], middlesizes[i][1], middlesizes[i][2], rnks[i][2]));
        end
        MPT(cores,mps.normcore)
    end