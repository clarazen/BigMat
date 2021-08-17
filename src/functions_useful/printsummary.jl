# Summary -----------------------------------------------------------------
macro Name(arg)
    string(arg)
end

function Base.show(io::IO, tt::TensorTrain)
    if     ndims(tt.cores[2]) == 3
        println(string("Summary of ",@Name(tt), " which is a vector (MPS/TT)"))
        rowdims = [size(tt,i,2) for i = 1:length(tt)]
    elseif ndims(tt.cores[2]) == 4
        println(string("Summary of ",@Name(tt), " which is a matrix (MPO/TTm)"))
        rowdims = [size(tt,i,2) for i = 1:length(tt)]
        coldims = [size(tt,i,3) for i = 1:length(tt)]
    else
        println(string("Summary of ",@Name(tt)))
    end
    println(string("Number of cores: ",Base.length(tt)))
    println(string("Ranks: \t\t ",LinearAlgebra.rank(tt::TensorTrain)))
    #println(string("Core sizes: \t ",size(tt)))
    println(string("Row dims: \t ",rowdims))
    println(string("Column dims:\t ",coldims))
end