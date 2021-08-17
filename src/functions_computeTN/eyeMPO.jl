function eye(middlesizes::Matrix{Int64})
    cores = Vector{Array{Float64,4}}(undef,order);
    for i = 1:size(middlesizes,1)
        core = Matrix(I,(middlesizes[i,1],middlesizes[i,2]));
        core = reshape(core, (1, middlesizes[i,1],middlesizes[i,2], 1) );
        cores[i] = convert(Array{Float64}, core);
    end    

    MPT(cores)
end