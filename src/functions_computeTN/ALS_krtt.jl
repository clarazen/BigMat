function ALS_krtt(y::Vector,kr::Vector{Matrix},rnks::Vector{Int},maxiter,λ::Float64)
    # computes components of tt from y = kr*tt with ALS:
    # tt is first split into the to be updates component and the rest U, 
    # then U and kr are multiplied with each other giving Ũ
    # then y = Ũ*tt[d] is solved for tt[d]

    D     = size(kr,1)
    Md    = size(kr[1],2)
    N     = length(y)
    cores = Vector{Array{Float64,3}}(undef,D)
    for i = 1:D # creating initial tensor train
        tmp = rand(rnks[i]*Md, rnks[i+1])
        cores[i] = reshape(tmp,(rnks[i], Md, rnks[i+1]))
    end
    tt0      = MPT(cores)
    tt       = tt0
    cova     = Vector{Matrix}(undef,D)
    for iter = 1:maxiter
        for d = 1:D
            ttm     = getU(tt,d)   # works       
            U       = krtimesttm(kr,transpose(ttm)) # works
            tmp     = (U*U'+λ*Matrix(I,size(U,1),size(U,1)))
            tt[d]   = reshape(tmp\(U*y),size(tt0[d]))
            cova[d] = tmp\Matrix(I,length(tt[d]),length(tt[d]))
        end
    end
    return tt,cova
end

function getU(tt::MPT{3},d::Int)
    D           = order(tt)
    middlesizes = size(tt,true)
    M           = middlesizes[d]
    newms       = zeros(2,D)
    newms[1,:]  = middlesizes
    newms[2,:]  = ones(D)
    ttm         = mps2mpo(tt,Int.(newms))
    ttm[d]      = reshape(Matrix(I,(M,M)),(1,M,M,1))
    if d>1
        ttm[d-1] = permutedims(ttm[d-1],(1,2,4,3))
    end
    if d<D
        ttm[d+1] = permutedims(ttm[d+1],(3,2,1,4))
    end

    return ttm
end

function kr2mat(Φ::Vector{Matrix})
    Φ_mat = ones(size(Φ[1],1),1)
    for d = size(Φ,1):-1:1
        Φ_mat = KhatriRao(Φ_mat,Φ[d],1)
    end    
    return Φ_mat
end