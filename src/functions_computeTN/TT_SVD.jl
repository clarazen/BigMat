function TT_SVD(tensor::Array{Float64},ϵ::Float64)
########################################################################    
#   Computes the cores of a TT for the given tensor and accuracy (acc)     
#   Resources:
#   V. Oseledets: Tensor-Train Decomposition, 2011, p.2301: Algorithm 1
#   April 2021, Clara Menzen
########################################################################
    N = ndims(tensor);
    cores = Vector{Array{Float64,3}}(undef,N);
    frobnorm = norm(tensor); 

    δ = ϵ / sqrt(N-1) * frobnorm;
    err2 = 0;
    rprev = 1;
    sizes = size(tensor);
    C = reshape( tensor, (sizes[1], Int(length(tensor) / sizes[1]) ));
    for k = 1 : N-1
        # truncated svd 
        F   = svd!(C); 
        rcurr = length(F.S);

        sv2 = cumsum(reverse(F.S).^2);
        tr  = Int(findfirst(sv2 .> δ^2))-1;
        if tr > 0
            rcurr = length(F.S) - tr;
            err2 += sv2[tr];
        end
        
        # new core
        cores[k] = reshape(F.U[:,1:rcurr],(rprev,sizes[k],rcurr));
        rprev    = rcurr;
        C        = Diagonal(F.S[1:rcurr])*F.Vt[1:rcurr,:];
        C        = reshape(C,(rcurr*sizes[k+1], Int(length(C) / (rcurr*sizes[k+1])) ) );
    end
    cores[N] = reshape(C,(rprev,sizes[N],1));
    return MPT(cores,N), sqrt(err2)/frobnorm
end

function TT_SVD(tensor::Array{Float64},d::Int,ϵ::Float64)
    ########################################################################    
    # for a symmetric matrix
        cores    = Vector{Array{Float64,3}}(undef,d);
        S        = zeros(2)
        frobnorm = norm(tensor); 
    
        δ     = ϵ / sqrt(d-1) * frobnorm;
        err2  = 0;
        rprev = 1;
        sizes = size(tensor);
        C     = reshape( tensor, (sizes[1], Int(length(tensor) / sizes[1]) ));
        for k = 1 : D
            # truncated svd 
            F   = svd!(C); 
            rcurr = length(F.S);
    
            sv2 = cumsum(reverse(F.S).^2);
            tr  = Int(findfirst(sv2 .> δ^2))-1;
            if tr > 0
                rcurr = length(F.S) - tr;
                err2 += sv2[tr];
            end
            
            # new core
            
            cores[k] = reshape(F.U[:,1:rcurr],(rprev,sizes[k],rcurr));
            rprev    = rcurr;
            C        = Diagonal(F.S[1:rcurr])*F.Vt[1:rcurr,:];
            C        = reshape(C,(rcurr*sizes[k+1], Int(length(C) / (rcurr*sizes[k+1])) ) );
            S        = F.S[1:rcurr]
        end
        return MPT(cores,D), S, sqrt(err2)/frobnorm
    end
    