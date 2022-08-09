function roundTT(mpt::MPT, ϵ::Float64)

    N           = order(mpt);
    cores       = Vector{Array{Float64,3}}(undef,N);
    middlesizes = size(mpt);
    frobnorm    = norm(mpt); 
    err2        = 0;
    
    # Right-to-left orthogonalization
        G = mpt[N];
        for k = N:-1:2
            G        = unfold(G,[1],"left");
            F        = qr(G');
            Q        = Matrix(F.Q);
            R        = Matrix(F.R);
            G        = reshape(Q', ( size(Q,2), middlesizes[k][1], Int(length(Q)/(size(Q,2)*middlesizes[k][1])) ) );
            cores[k] = G;
            G        = nmodeproduct(R,mpt[k-1],3);
        end
        cores[1] = G;
        # Compression of the orthogonalized representation
        δ = ϵ / sqrt(N-1) * frobnorm;
        for k = 1:N-1
            F     = svd(unfold(cores[k],[3],"right"),full=false);
            rcurr = length(F.S);
            sv2   = cumsum(reverse(F.S).^2);
            tr    = Int(findfirst(sv2 .> δ^2))-1;
            if tr > 0
                rcurr = length(F.S) - tr;
                err2 += sv2[tr];
            end
            Utr = F.U[:,1:rcurr];
            Str = F.S[1:rcurr];
            Vtr = F.Vt[1:rcurr,:];

            cores[k]   = reshape( Utr,( Int(length(Utr)/(size(cores[k],2)*size(Vtr,1))), size(cores[k],2), size(Vtr,1) ) );
            cores[k+1] = nmodeproduct(Matrix((Vtr'*Diagonal(Str))'),cores[k+1],1);     
        end
        return MPT(cores);
    end

    function roundTT(ttm::MPT{4}, ϵ::Float64)
        tt = mpo2mps(ttm);
        tt = roundTT(tt, ϵ)
        return mps2mpo(tt,size(ttm,true))
    end