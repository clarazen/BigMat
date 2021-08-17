function roundTT(mpt::MPT, acc::Float64)

    order       = ndims(mpt);
    cores       = Vector{Array{Float64,3}}(undef,order);
    rnks        = ranks(mpt);
    middlesizes = size(mpt);
    
    # Right-to-left orthogonalization
        G = mpt[order];
        for k = order:-1:2
            G = unfold(G,[1],"left");
            F = qr(G');
            Q = Matrix(F.Q);
            R = Matrix(F.R);
            G   = reshape(Q', ( size(Q,2), middlesizes[k][1], Int(length(Q)/(size(Q,2)*middlesizes[k][1])) ) );
            cores[k] = G;
            G = contractmodes(mpt[k-1],Matrix(R'),[3],[1]);
        end
        cores[1] = G;
        # Compression of the orthogonalized representation
        delta = acc / sqrt(order-1) * norm(mpt);
         for k = 1:order-1
            F   = svd(unfold(cores[k],[3],"right"),full=false);
            err = norm(F.S[end]);
            svrem = length(F.S);
            while err <= delta && svrem > 1
                svrem = svrem-1;
                err = norm(F.S[svrem+1:end]);
            end
            Utr = F.U[:,1:svrem];
            Str = F.S[1:svrem];
            Vtr = F.Vt[1:svrem,:];

            cores[k]   = reshape( Utr,( Int(length(Utr)/(size(cores[k],2)*size(Vtr,1))), size(cores[k],2), size(Vtr,1) ) );
            cores[k+1] = contractmodes(Matrix((Vtr'*Diagonal(Str))'),cores[k+1],[2],[1]);     
        end
        mptr = MPT(cores);
    end