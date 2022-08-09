# ALS for matrices without initial TT
function TTm_ALS(matrix::Matrix,middlesizes::Matrix{Int64},rnks::Vector{Int64})
    tensor  = reshape(matrix,(middlesizes[1,:]...,middlesizes[2,:]...));
    d       = length(rnks)-1
    permind = Tuple([ (i-1)*d+j for j in 1:d for i in 1:2 ]);
    tensor  = permutedims(tensor,permind);
    resind  = Tuple([prod(col) for col in eachcol(middlesizes)]);
    tensor  = reshape(tensor,resind)
    tt      = TT_ALS(tensor,rnks);
    return MPT( [reshape(tt[i],(rnks[i], middlesizes[:,i]..., rnks[i+1])) for i = 1:order(tt)],d );
end

# ALS for matrices with initial TT
function TTm_ALS(matrix::Matrix,ttm0::MPT{4})
    d           = order(ttm0)
    middlesizes = [[size(ttm0)[i][1] for i = 1:d]'; [size(ttm0)[i][2] for i = 1:d]'];
    ressz       = ([size(ttm0)[i][1] for i = 1:d]'..., [size(ttm0)[i][2] for i = 1:d]'... )
    tensor      = reshape(matrix,ressz);
    permind     = Tuple([ (i-1)*d+j for j in 1:d for i in 1:2 ]);
    tensor      = permutedims(tensor,permind);
    resind      = Tuple([prod(col) for col in eachcol(middlesizes)]);
    tensor      = reshape(tensor,resind)
    tt0         = mpo2mps(ttm0)
    tt          = TT_ALS(tensor,tt0);
    rnks        = rank(tt);
    return MPT( [reshape(tt[i],(rnks[i][1], middlesizes[i][1], middlesizes[i][1], rnks[i][2])) for i = 1:d],d );
end

# ALS for symmetric matrices
function TTm_ALS(symmat::Symmetric{Float64, Matrix{Float64}},middlesizes::Matrix{Int64},rnks::Vector{Int64})
    d     = length(rnks)-1
    cores = Vector{Array{Float64,4}}(undef,d);
    R     = Matrix{Float64}
    for i = 1:d
        cores[i] = zeros(rnks[i],middlesizes[1,i],middlesizes[2,i],rnks[i+1])
        for rleft = 1:rnks[i]
            for rright = 1:rnks[i+1]
                tmp = randn(middlesizes[1,i],middlesizes[2,i])
                cores[i][rleft,:,:,rright] = tmp*tmp'
            end
        end
        tmp      = reshape(cores[i],(rnks[i]*middlesizes[1,i]*middlesizes[2,i],rnks[i+1]))
        F        = qr(tmp)
        cores[i] = reshape(Matrix(F.Q),(rnks[i],middlesizes[1,i],middlesizes[2,i],rnks[i+1]))
        if i>1
            cores[i] = nmodeproduct(R,cores[i],1)
        end
        R = F.R
    end
    ttm0 = MPT(cores,d)
    return TTm_ALS(Matrix(symmat),ttm0)
end

# ALS for diagonal matrices
function TTm_ALS(diag::Vector,middlesizes::Matrix,rnks::Vector)
    d      = length(rnks)-1
    tensor = reshape(diag,Tuple(middlesizes[1,:]))
    tt     = TT_ALS(tensor,rnks)
    N      = size(middlesizes,2)
    cores  = Vector{Array{Float64,4}}(undef,N)
    for i = 1:N
        cores[i] = zeros(rnks[i],middlesizes[i],middlesizes[i],rnks[i+1])
        for ri = 1:rnks[i]
            for rii = 1:rnks[i+1]
                cores[i][ri,:,:,rii] = Matrix(Diagonal(tt[i][ri,:,rii]))
            end
        end
    end
    return MPT(cores,d)
end
