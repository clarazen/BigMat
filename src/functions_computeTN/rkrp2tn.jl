function rkrp2tn(Ũ::Vector{Matrix{Float64}})
    # repeated Kathti-Rao product into tensor network
    # Source: Batselier et al.: Tensor network subspace identification of polynomial state space models, Algorithm 2
    
    N = length(Ũ);
    M = size(Ũ[1],1);
    P = size(Ũ[1],2);
    U = Vector{Array{Float64,4}}(undef,N);
    U[1] = reshape(Ũ[N],(1,M,P,1));
    r1 = 1;
    for i = 1:N-1
        T = reshape(U[i],(r1*M,P));
        Ttmp = zeros(r1*M^2,P);
        @inbounds @simd for j = 1:P
            @views kron!(Ttmp[:,j],Ũ[N-i][:,j],T[:,j])
        end

        T = reshape(Ttmp,(r1*M,M*P));
        F = svd!(T);
        r2 = length(F.S);
        U[i] = reshape(F.U,(r1,M,1,r2));
        U[i+1] = reshape(diagm(F.S)*F.Vt,(r2,M,P,1));
        r1 = r2;
    end
    return MPT(U)
end