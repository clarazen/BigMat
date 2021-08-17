# different calls, either with maxiter or residual change under a certain value (maxiter default in case does not converge)
function TT_ALS(tensor::Array{Float64},rnks::Vector{Int64})
    maxiter = 1;
    N     = ndims(tensor);
    sizes = size(tensor);
    cores = Vector{Array{Float64,3}}(undef,N);
    res   = Vector{Float64}();
    for i = 1:N-1 # creating site-N canonical initial tensor train
        tmp = qr(rand(rnks[i]*sizes[i], rnks[i+1]));
        cores[i] = reshape(Matrix(tmp.Q),(rnks[i], sizes[i], rnks[i+1]));
    end
    cores[N] = reshape(rand(rnks[N]*sizes[N]),(rnks[N], sizes[N], 1));
    tt = MPT(cores);

    for i = 1:maxiter
        # update last core   
        Gleft    = supercores(tt,N);
        newsizes = (prod(sizes[1:N-1]), sizes[N]);
        tmp      = Gleft*reshape(tensor,Tuple(newsizes));
        tt[N]    = reshape(tmp,(rnks[N],sizes[N],1));
        ttNvec   = reshape(tmp,(length(tt[N]), 1));
        err      = norm(kron(Matrix(I,sizes[N],sizes[N]),Gleft')*ttNvec-tensor[:])/norm(tensor[:]);
        push!(res,err)
        shiftMPTnorm(tt,N,-1);
        
        # update middle cores from right to left
        for n = N-1:-1:2
            Gleft, Gright = supercores(tt,n);
            newsizes1     = (prod(sizes[1:n-1]), prod(sizes[n:N]));
            tmp           = Gleft*reshape(tensor,newsizes1);
            newsizes2     = (rnks[n]*sizes[n], prod(sizes[n+1:N]));
            tmp           = reshape(tmp,newsizes2)*Gright;
            tt[n]         = reshape(tmp,(rnks[n],sizes[n],rnks[n+1]));
            ttnvec        = reshape(tmp,(length(tt[n]), 1));
            err           = norm(kron(Gright,kron(Matrix(I,sizes[n],sizes[n]),Gleft'))*ttnvec-tensor[:])/norm(tensor[:]);
            push!(res,err)
            shiftMPTnorm(tt,n,-1);
        end

        # update first core
        Gright   = supercores(tt,1);
        newsizes = (sizes[1], prod(sizes[2:N]));
        tmp      = reshape(tensor,Tuple(newsizes))*Gright;
        tt[1]    = reshape(tmp,(1,sizes[1],rnks[2]));
        tt1vec   = reshape(tmp,(length(tt[1]), 1));
        err      = norm(kron(Gright,Matrix(I,sizes[1],sizes[1]))*tt1vec-tensor[:])/norm(tensor[:]);
        push!(res,err)
        shiftMPTnorm(tt,1,1);

        # update middle cores from left to right
        for n = 2:N-1
            Gleft, Gright = supercores(tt,n);
            newsizes1     = (prod(sizes[1:n-1]), prod(sizes[n:N]));
            tmp           = Gleft*reshape(tensor,newsizes1);
            newsizes2     = (rnks[n]*sizes[n], prod(sizes[n+1:N]));
            tmp           = reshape(tmp,newsizes2)*Gright;
            tt[n]         = reshape(tmp,(rnks[n],sizes[n],rnks[n+1]));
            ttnvec        = reshape(tmp,(length(tt[n]), 1));
            err           = norm(kron(Gright,kron(Matrix(I,sizes[n],sizes[n]),Gleft'))*ttnvec-tensor[:])/norm(tensor[:]);
            push!(res,err)
            shiftMPTnorm(tt,n,1);
        end
    end
    return tt, res
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

        mpt[n]     = reshape(Q, size(mpt[n]));
        mpt[n+dir] = nmodeproduct(R,mpt[n+dir],ind);
end