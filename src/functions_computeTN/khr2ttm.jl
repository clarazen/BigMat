function khr2ttm(Ũ::Vector{Matrix{Float64}},ϵ::Float64)
    # repeated Kathti-Rao product into tensor train matrix
    # Source: Batselier et al.: Tensor network subspace identification of polynomial state space models, Algorithm 2
    
    N    = length(Ũ);
    M    = size(Ũ[1],1);
    P    = size(Ũ[1],2);
    U    = Vector{Array{Float64,4}}(undef,N);
    U[1] = reshape(Ũ[N],(1,M,P,1));
    r1   = 1; err2 = 0;
    δ    = ϵ / sqrt(N-1) * norm(.*(Ũ));
    for i = 1:N-1
        T = reshape(U[i],(r1*M,P));
        Ttmp = KhatriRao(Ũ[N-i],T,2)

        T = reshape(Ttmp,(r1*M,M*P));
        F = svd!(T);
        r2 = length(F.S);
        sv2 = cumsum(reverse(F.S).^2);
        tr  = findfirst(sv2 .> δ^2);
        if typeof(tr) == Nothing
            tr = length(sv2)-1;
        else
            tr = Int(tr)-1;
        end
        if tr > 0
            r2 = r2 - tr;
            err2 += sv2[tr];
        end

        U[i] = reshape(F.U[:,1:r2],(r1,M,1,r2));
        U[i+1] = reshape(diagm(F.S[1:r2])*F.Vt[1:r2,:],(r2,M,P,1));
        r1 = r2;
    end
    return MPT(U), err2
end

