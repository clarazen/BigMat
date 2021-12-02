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
    tt = MPT(cores,N);
    return TT_ALS(tensor,rnks,tt)
end

# with initial tt
function TT_ALS(tensor::Array{Float64},rnks::Vector{Int64},tt::MPS)
    return TT_ALS(tensor,rnks,tt,tt.normcore)
end

# with / without orthogonalization
function TT_ALS(tensor::Array{Float64},rnks::Vector{Int64},tt::MPS,normcore::Int64)
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
                swipe = [collect(tt.normcore:N)..., collect(N-1:-1:2)..., collect(1:tt.normcore-1)...];
                n     = swipe[k];
                if tt.normcore == 1
                    Dir = Int.([ones(1,N-1)..., -ones(1,N-1)...]); 
                elseif tt.normcore == N
                    Dir = Int.([-ones(1,N-1)...,ones(1,N-1)...]);
                else
                    Dir = Int.([ones(1,N-tt.normcore)...,-ones(1,N-1)...,ones(1,tt.normcore-1)...]);
                end
                UTy   = getUTy(tt,tensor,n);
                tt[n] = reshape(UTy,(rnks[n][1],sizes[n][1],rnks[n][2]));
                shiftMPTnorm(tt,n,Dir[k])
            end
        end
    end
    return tt
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

    return kron(kron(Gright, Matrix(I,sizes[n][1],sizes[n][1])), Gleft)
end