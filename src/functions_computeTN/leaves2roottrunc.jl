function leaves2roottrunc(tensor::Array{Float64},p::Int64,ranks::Vector{Int64})
    # Source: Algorithm 2 of Hierarchical SVD of Tensors by Lars Grasedyck
    N  = ndims(tensor);
    sz = size(tensor);
    Ut = Matrix{Float64};
    Cp = unfold(tensor,[1],"left");
    for t = 1:N
        At = Cp;
        F  = svd!(At); 
        Ut = F.U[:,1:ranks[t]];
        Cp = Ut'*Cp;
        Cp = reshape(Cp,(ranks[1:t]..., sz[t+1:end]...));
        if t<N
            Cp = unfold(Cp,[t+1],"left");
        end
    end
    
    for l = p-1:-1:1
        
    end

end