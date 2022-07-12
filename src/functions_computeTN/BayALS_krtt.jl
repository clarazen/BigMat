function ALS_krtt(y::Vector,kr::Vector{Matrix{Float64}},rnks::Vector{Int},ϵ::Float64,maxiter::Int,m0::Vector{Vector},P0::Vector{Matrix})
    # computes componente of tt from y = kr*tt with ALS:
    # tt is first split into the to be updates component and the rest U, 
    # then U and kr are multiplied with each other giving Ũ
    # then y = Ũ*tt[d] is solved for tt[d]

    D     = size(kr,1)
    Md    = size(kr[1],2)
    cores = Vector{Array{Float64,3}}(undef,D)
    for i = 1:D
        cores[i] = reshape(m0,(rnks[i], Md, rnks[i+1]))
    end
    tt    = MPT(cores,0)
    invP0 = inv(P0)
    m     = Vector{Vector}(undef,D)
    P     = Vector{Matrix}(undef,D)
    
    for iter = 1:maxiter
        for d = 1:D
            ttm   = getU(tt,d)
            # make krtimesttm without the rounding step, but resulting in a matrix
            U     = mpo2mat(krtimesttm(kr,transpose(ttm),ϵ))
                
            P[d]  = inv(invP0 + U*U'/σ_n^2)
            m[d]  = P[d]*(U*y/σ_n^2 + invP0*m0)
            tt[d] = reshape(m0,size(tt0[d]))
        end
    end
    return m,P
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