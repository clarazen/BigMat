function ALS_krtt(y::Vector,kr::Vector{Matrix},rnks::Vector{Int},maxiter)
    # computes componente of tt from y = kr*tt with ALS:
    # tt is first split into the to be updates component and the rest U, 
    # then U and kr are multiplied with each other giving Ũ
    # then y = Ũ*tt[d] is solved for tt[d]

    D     = size(kr,1)
    Md    = size(kr[1],2)
    cores = Vector{Array{Float64,3}}(undef,D)
    for i = 1:D-1 # creating site-N canonical initial tensor train
        tmp = qr(rand(rnks[i]*Md, rnks[i+1]))
        cores[i] = reshape(Matrix(tmp.Q),(rnks[i], Md, rnks[i+1]))
    end
    cores[D] = reshape(rand(rnks[D]*Md),(rnks[D], Md, 1))
    tt0      = MPT(cores,D)
    tt       = tt0
    cova     = Vector{Matrix}(undef,D)
    for iter = 1:maxiter
        for d = 1:D
            ttm     = getU(tt,d)
            U       = krtimesttm(kr,transpose(ttm))
            UtU     = U*U'
            Uty     = U*y
            tt[d]   = reshape(UtU\Uty,size(tt0[d]))
            cova[d] = UtU\Matrix(I,length(tt[d]),length(tt[d]))
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