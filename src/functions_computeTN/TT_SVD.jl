function TT_SVD(tensor::Array{Float64},ϵ::Float64)
########################################################################    
#   Computes the cores of a TT for the given tensor and accuracy (acc)     
#   Resources:
#   V. Oseledets: Tensor-Train Decomposition, 2011, p.2301: Algorithm 1
#   April 2021, Clara Menzen
########################################################################
    N = ndims(tensor);
    cores = Vector{Array{Float64,3}}(undef,N);
     
    δ = ϵ / sqrt(N-1) * norm(tensor);
    rprev = 1;
    sizes = size(tensor);
    C = reshape( tensor, (sizes[1], Int(length(tensor) / sizes[1]) ));
    for k = 1 : N-1
        # truncated svd 
        F   = svd!(C);
        err = norm(F.S[end]);
        rcurr = length(F.S);
        while err <= δ && rcurr > 1
            rcurr = rcurr-1;
            err = norm(F.S[rcurr+1:end]);
        end
        # new core
        cores[k] = reshape(F.U[:,1:rcurr],(rprev,sizes[k],rcurr));
        rprev    = rcurr;
        C        = Diagonal(F.S[1:rcurr])*F.Vt[1:rcurr,:];
        C        = reshape(C,(rcurr*sizes[k+1], Int(length(C) / (rcurr*sizes[k+1])) ) );
    end
    cores[N] = reshape(C,(rprev,sizes[N],1));
    MPT(cores);
end

