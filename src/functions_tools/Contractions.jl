    function contractmodes(tensor1::Array{Float64},tensor2::Array{Float64},ind_contract::Matrix{Int64},ind_rem::Matrix{Int64})
        # function that contracts modes specified in ind_contract, combines indices specified in ind_comn and leaves indices free specified in ind_free
        # contraction
        matrix1  = unfold(tensor1,ind_contract[:,1],"right");
        matrix2  = unfold(tensor2,ind_contract[:,2],"left");
        matrix   = matrix1*matrix2;
        sizes_1  = deleteat!(collect(size(tensor1)),sort(ind_contract[:,1]));
        sizes_2  = deleteat!(collect(size(tensor2)),sort(ind_contract[:,2]));
        newsizes = vcat(sizes_1,sizes_2);
        tensor   = reshape(matrix,newsizes...);
        
        # combining specified modes and freeing specified modes
        dimrem_1 = ndims(tensor1) - size(ind_contract,1);
        dimrem_2 = ndims(tensor2) - size(ind_contract,1) 
        newperm  = Vector{Int64}(undef,dimrem_1+dimrem_2);
        for i = 1:size(ind_contract,1)
            for j = 1:size(ind_rem,1)
                if ind_contract[i,1] < ind_rem[j,1]
                    ind_rem[j,1] = ind_rem[j,1]-1;
                end
                if ind_contract[i,2] < ind_rem[j,2]
                    ind_rem[j,2] = ind_rem[j,2]-1;
                end
            end
        end 
        k = 1;
        for i = 1:size(ind_rem,1)
            if ind_rem[i,1] > 0
                newperm[k] = ind_rem[i,1];
                k = k+1;
            end
            if ind_rem[i,2] > 0
                newperm[k] = ind_rem[i,2]+dimrem_1;
                k = k+1;
            end
        end
        tensor   = permutedims(tensor,Tuple(newperm));

        newsizes = Int64[];
        for i = 1:size(ind_rem,1)
            if ind_rem[i,1] > 0
                size_1 = sizes_1[ind_rem[i,1]];
            else
                size_1 = 1;
            end
            if ind_rem[i,2] > 0
                size_2 = sizes_2[ind_rem[i,2]];
            else
                size_2 = 1;
            end
            push!(newsizes,size_1*size_2);
        end
        reshape(tensor,newsizes...);
    end

    function contractmodes(tensor1::Array{Float64},tensor2::Array{Float64},ind_contract::Matrix{Int64})
        # contract specified indices and free the remaining indices
        matrix1  = unfold(tensor1,ind_contract[:,1],"right")
        matrix2  = unfold(tensor2,ind_contract[:,2],"left")
        matrix   = matrix1*matrix2
        sizes_1  = deleteat!(collect(size(tensor1)),sort(ind_contract[:,1]));
        sizes_2  = deleteat!(collect(size(tensor2)),sort(ind_contract[:,2]));
        newsizes = vcat(sizes_1,sizes_2);
        reshape(matrix,newsizes...);
    end

    function contractcores(core1::Array{Float64,4},core2::Array{Float64,3})
        contractmodes(core1,core2,[3 2],[1 1; 2 0; 4 3]);
    end

    function contractcores(core1::Array{Float64,4},core2::Array{Float64,4})
        contractmodes(core1,core2,[3 2],[1 1; 2 0; 0 3; 4 4]);
    end

    function contractcores(core1::Array{Float64,3},core2::Array{Float64,3})
        contractmodes(core1,core2,[2 2],[1 1; 3 3]);
    end

    function contractcores(core1::Array{Float64,3},core2::Array{Float64,4})
        contractmodes(core1,core2,[2 2],[1 1; 0 3; 3 4]);
    end

    function nmodeproduct(matrix::Matrix{Float64},tensor::Array{Float64},n::Int64)
    #        ___         ___
    #    ___/ A \_______/ X \_________________________
    #     J \___/  I_n  \___/ I_1...I_n-1 I_n+1...I_N
    #
    #              ____
    #             /XtnA\
    #    =====>   \____/
    #             /||||\
    #           I_1  J  I_N 
    # Kolda, Bader: Tensor Decomposition and Applications, 2009. Page 460

        tensor = contractmodes(matrix,tensor,[2 n]); # contract mode n and leave all other modes free
        permutedims(tensor,[collect(2:n)..., 1, collect(n+1:ndims(tensor))...]) 
    end

    function nmodeproduct(vector::Vector,tensor::Array,n::Int64)
        tensor = contractmodes(reshape(vector,(1,length(vector))),tensor,[2 n]); # contract mode n and leave all other modes free
        permutedims(tensor,[collect(2:n)..., 1, collect(n+1:ndims(tensor))...])
    end

    function vectorbymatrix(A::MPT{4},b::MPT{3})
        d = order(A)
        vec = contractcores(b[1],A[1])[1,:,:]
        for i = 2:d-1
            vec = vec * contractcores(b[i],A[i])[:,1,:]
        end
        return vec = vec * contractcores(b[d],A[d])[:,:,1]
    end

    function matrixbyvector(A::MPT{4},b::MPT{3})
        d = order(A)
        vec = contractcores(b[1],A[1])[1,:,:]
        for i = 2:d-1
            vec = vec * contractcores(b[i],A[i])[:,1,:]
        end
        return vec = (vec * contractcores(b[d],A[d])[:,:,1])'
    end

    # computation of vector represented by an MPS
    function mps2vec(mps::MPT{3})
        tensor = unfold(mps[1],[3],"right");
        for i = 2:order(mps)
            tensor = tensor*unfold(mps[i],[1],"left");
            tensor = reshape(tensor, (Int(length(tensor)/size(mps[i],3)), size(mps[i],3)));
        end
        vector = tensor;
        return vector
    end

    # computation of matrix represented by an MPO
    function mpo2mat(mpo::MPT{4})
        tensor = unfold(mpo[1],[4],"right");
        sizes = size(mpo);
        newsizes = [sizes[1]...];
        matsize1 = sizes[1][1];
        matsize2 = sizes[1][2];
        for i = 2:order(mpo)
            tensor = tensor*unfold(mpo[i],[1],"left");
            tensor = reshape(tensor, (Int(length(tensor)/size(mpo[i],4)), size(mpo[i],4)));
            newsizes = (newsizes..., sizes[i]...);
            matsize1 = matsize1*sizes[i][1];
            matsize2 = matsize2*sizes[i][2];
        end
        tensor = reshape(tensor,newsizes);
        tensor = permutedims(tensor,[collect(1:2:length(newsizes)-1)..., collect(2:2:length(newsizes))...]);
        matrix = reshape(tensor,(matsize1,matsize2));
        return matrix
    end