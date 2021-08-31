function eye(middlesizes::Matrix{Int64})

    cores = Vector{Array{Float64,4}}(undef,size(middlesizes,2));
    for i = 1:size(middlesizes,2)
        core = Matrix(I,(middlesizes[1,i],middlesizes[2,i]));
        core = reshape(core, (1, middlesizes[1,i],middlesizes[2,i], 1) );
        cores[i] = convert(Array{Float64}, core);
    end    

    MPT(cores)
end