# Comments:
# ALS without orthog is really slow (probably getUTU needs to be optimized) 
# and almost never used unless an initial tt is inputted which is not site-k

# no initial tt, automatically with orthogonalization
function TT_ALS(tensor::Array{Float64},rnks::Vector{Int64})
    N     = ndims(tensor);
    sizes = size(tensor);
    cores = Vector{Array{Float64,3}}(undef,N);
    for i = 1:N-1 # creating site-N canonical initial tensor train
        tmp = qr(rand(rnks[i]*sizes[i], rnks[i+1]));
        cores[i] = reshape(Matrix(tmp.Q),(rnks[i], sizes[i], rnks[i+1]));
    end
    cores[N] = reshape(rand(rnks[N]*sizes[N]),(rnks[N], sizes[N], 1));
    tt0 = MPT(cores,N);
    return TT_ALS(tensor,tt0)
end

function TT_ALS(vector::SparseVector{Float64, Int64},sizes::Vector,rnks::Vector{Int64})
    D     = length(sizes);
    cores = Vector{Array{Float64,3}}(undef,D);
    for i = 1:D-1 # creating site-N canonical initial tensor train
        tmp = qr(rand(rnks[i]*sizes[i], rnks[i+1]));
        cores[i] = reshape(Matrix(tmp.Q),(rnks[i], sizes[i], rnks[i+1]));
    end
    cores[D] = reshape(rand(rnks[D]*sizes[D]),(rnks[D], sizes[D], 1));
    tt0 = MPT(cores,D);
    return TT_ALS(vector,sizes,tt0)
end

# with / without orthogonalization
function TT_ALS(tensor::Array{Float64},tt::MPS)
    maxiter = 10;
    N       = order(tt);
    rnks    = rank(tt);
    sizes   = size(tt);

    for i = 1:maxiter
        for k = 1:2N-2
            if tt.normcore == 0
                swipe = [collect(1:N)..., collect(N-1:-1:2)...];
                n     = swipe[k];
                UTU   = getUTU(tt,n);
                UTy   = getUTy(tt,tensor,n);
                tt[n] = reshape(inv(UTU)*UTy,(rnks[n][1],sizes[n][1],rnks[n][2]));
            else
                swipe = [collect(N:-1:2)..., collect(1:N-1)...];
                Dir   = Int.([-ones(1,N-1)...,ones(1,N-1)...]);
                n     = swipe[k];
                UTy   = getUTy(tt,tensor,n);
                tt[n] = reshape(UTy,(rnks[n][1],sizes[n][1],rnks[n][2]));
                shiftMPTnorm(tt,n,Dir[k])
            end
        end
    end
    return tt
end

# with / without orthogonalization
function TT_ALS(vector::SparseVector{Float64, Int64},sz::Vector{Int},tt::MPS)
    maxiter = 10;
    N       = order(tt);
    rnks    = rank(tt);
    sizes   = size(tt);

    for i = 1:maxiter
        for k = 1:2N-2
            if tt.normcore == 0
                swipe = [collect(1:N)..., collect(N-1:-1:2)...];
                n     = swipe[k];
                UTU   = getUTU(tt,n);
                UTy   = getUTy(tt,vector,sz,n);
                tt[n] = reshape(inv(UTU)*UTy,(rnks[n][1],sizes[n][1],rnks[n][2]));
            else
                swipe = [collect(N:-1:2)..., collect(1:N-1)...];
                Dir   = Int.([-ones(1,N-1)...,ones(1,N-1)...]);
                n     = swipe[k];
                UTy   = getUTy(tt,vector,sz,n);
                tt[n] = reshape(UTy,(rnks[n][1],sizes[n][1],rnks[n][2]));
                shiftMPTnorm(tt,n,Dir[k])
            end
        end
    end
    return tt
end

# TT-ALS for a vector without initial tt
function TT_ALS(vector::Vector{Float64},middlesizes::Matrix{Int64},rnks::Vector{Int64})
    tensor = reshape(vector,Tuple(middlesizes));
    return TT_ALS(tensor,rnks);
end


# TT-ALS for vector with initial TT
function TT_ALS(vector::Vector{Float64},tt0::MPT)
    tensor = reshape(vector,Tuple([size(tt0)[i][1] for i = 1:order(tt0)]));
    return TT_ALS(tensor,tt0);  
end


##########################################
# functions for ALS with orthogonalization
function getUTy(tt::MPS,tensor::Array{Float64},n::Int64)
    N     = order(tt);
    sizes = size(tensor);
    rnks  = rank(tt);
    if n == N 
        Gleft    = supercores(tt,N);
        newsizes = (prod(sizes[1:N-1]), sizes[N]);
        UTy      = Gleft*reshape(tensor,Tuple(newsizes));
    elseif n == 1
        Gright   = supercores(tt,1);
        newsizes = (sizes[1], prod(sizes[2:N]));
        UTy      = reshape(tensor,Tuple(newsizes))*Gright;
    else
        Gleft, Gright = supercores(tt,n);
        newsizes1     = (prod(sizes[1:n-1]), prod(sizes[n:N]));
        tmp           = Gleft*reshape(tensor,newsizes1);
        newsizes2     = (rnks[n][1]*sizes[n], prod(sizes[n+1:N]));
        UTy           = reshape(tmp,newsizes2)*Gright;
    end
    return UTy[:]
end

function getUTy(tt::MPS,vector::SparseVector{Float64, Int64},sz::Vector{Int},n::Int64)
    N     = order(tt);
    sizes = sz
    rnks  = rank(tt);
    if n == N 
        Gleft    = supercores(tt,N);
        newsizes = (prod(sizes[1:N-1]), sizes[N]);
        UTy      = Gleft*reshape(vector,Tuple(newsizes));
    elseif n == 1
        Gright   = supercores(tt,1);
        newsizes = (sizes[1], prod(sizes[2:N]));
        UTy      = reshape(vector,Tuple(newsizes))*Gright;
    else
        Gleft, Gright = supercores(tt,n);
        newsizes1     = (prod(sizes[1:n-1]), prod(sizes[n:N]));
        tmp           = Gleft*reshape(vector,newsizes1);
        newsizes2     = (rnks[n][1]*sizes[n], prod(sizes[n+1:N]));
        UTy           = reshape(tmp,newsizes2)*Gright;
    end
    return UTy[:]
end

function supercores(tt::MPT{3},n::Int64)
    N     = order(tt);
    sizes = size(tt);
    rnks  = rank(tt);
    if  n == 1
        Gright = tt[2];
        sright = sizes[2][1];
        for i = 3:N
            Gright = Gright*tt[i];
            sright = sright*sizes[i][1];
        end
        return Matrix(reshape(Gright,(rnks[1][2], sright))')
    elseif n == N
        Gleft = tt[1];
        sright = sizes[1][1];
        for i = 2:N-1
            Gleft = Gleft*tt[i];
            sright = sright*sizes[i][1];
        end
        return Matrix(reshape(Gleft,(sright, rnks[N][1]))')
    else
        Gleft = tt[1];
        sleft = sizes[1][1];
        for i = 2:n-1
            Gleft = Gleft*tt[i];
            sleft = sleft*sizes[i][1]
        end 
        Gright = tt[n+1];
        sright = sizes[n+1][1];
        for i = n+2:N
            Gright = Gright*tt[i];
            sright = sright*sizes[i][1];
        end
        return Matrix(reshape(Gleft,(sleft, rnks[n][1]))'), Matrix(reshape(Gright,(rnks[n][2],sright))')
    end
end

function shiftMPTnorm(mpt::MPT,n::Int64,dir::Int64)

        if dir == 1
            Gl = unfold(mpt[n],[ndims(mpt[n])],"right");
            ind   = 1;
            F = qr(Gl);
            R = Matrix(F.R); Q = Matrix(F.Q);
        elseif dir == -1
            Gr = unfold(mpt[n],[1],"left");
            ind  = 3;
            F = qr(Gr');
            R = Matrix(F.R); Qt = Matrix(F.Q);
            Q = Qt';
        end

        mpt[n]       = reshape(Q, size(mpt[n]));
        mpt[n+dir]   = nmodeproduct(R,mpt[n+dir],ind);
        mpt.normcore = mpt.normcore + dir; 
        return mpt
end

# function for ALS without orthogonalization
function getUTU(tt::MPS,n::Int64)
    N     = order(tt);
    sizes = size(tt);
    rnks  = rank(tt);

    Gleft = [1];
    for i = 1:n-1
        Gleft = Gleft * contractcores(tt[i],tt[i]);
    end
    Gleft = reshape(Gleft,(rnks[n][1],rnks[n][1]));

    Gright = [1];
    for i = N:-1:n+1
        Gright = contractcores(tt[i],tt[i]) * Gright;
    end
    Gright = reshape(Gright,(rnks[n][2],rnks[n][2]));

    return kron(kron(Gright, 1.0*Matrix(I,sizes[n][1],sizes[n][1])), Gleft)
end

function shiftMPTnorm(mpt::MPT,cova::Matrix,d::Int64,dir::Int64)

    if dir == 1
        Gl = unfold(mpt[d],[ndims(mpt[d])],"right");
        ind   = 1;
        F = qr(Gl);
        R = Matrix(F.R); Q = Matrix(F.Q);
    elseif dir == -1
        Gr = unfold(mpt[d],[1],"left");
        ind  = 3;
        F = qr(Gr');
        R = Matrix(F.R); Qt = Matrix(F.Q);
        Q = Qt';
    end

    mpt[d]       = reshape(Q, size(mpt[d]));
    mpt[d+dir]   = nmodeproduct(R,mpt[d+dir],ind);
    mpt.normcore = mpt.normcore + dir; 

    if dir == 1
        # take out part with norm from covariance matrix of updated TT-core
        newcova_o = covaorth1(mpt,cova,pinv(R'),pinv(R),d,1,0);
    elseif dir == -1
        # take out part with norm from covariance matrix of updated TT-core
        newcova_o = covaorth2(mpt,cova,pinv(R'),pinv(R),d,0,1);
    end

    return mpt,newcova_o

end

function covaorth1(ttm::MPT,P::Matrix,Rleft::Matrix,Rright::Matrix,d::Int,dir1::Int,dir2::Int)

    r   = rank(ttm,true);
    sY  = size(ttm,true);
        
    sP  = size(P);
    P   = reshape(P,(Int(length(P)/r[d+dir1]), r[d+dir1]))*Rright;
    P   = reshape(P,(r[d+dir2], sY[d+dir2], r[d+dir1], sP[2]));
    P   = permutedims(P,[3, 1, 2, 4]);
    P   = Rleft*reshape(P,(r[d+dir1], Int(length(P)/r[d+dir1]) ));
    P   = reshape(P,(r[d+dir1], r[d+dir2], sY[d+dir2], sP[2] ));
    P   = permutedims(P,[2, 3, 1, 4]);
    P   = reshape(P,(sP[1], sP[2]));
        
end
    
    
function covaorth2(ttm::MPT,P::Matrix,Rleft::Matrix,Rright::Matrix,d::Int,dir1::Int,dir2::Int)

    r   = rank(ttm,true);
    sY  = size(ttm,true);
            
    sP = size(P);
    P  = reshape(P,(sP[1], r[d+dir1], sY[d+dir1], r[d+dir2]));
    P  = permutedims(P,[1, 3, 4, 2]);
    P  = reshape(P,(Int(length(P)/r[d+dir1]), r[d+dir1] ))*Rright;
    P  = reshape(P,(sP[1], sY[d+dir1], r[d+dir2], r[d+dir1]));
    P  = permutedims(P,[1, 4, 2, 3]);
    P  = Rleft*reshape(P,(r[d+dir1], Int(length(P)/r[d+dir1])));
    P  = reshape(P,(sP[1], sP[2]));
            
end
        
            