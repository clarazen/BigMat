@testset "TensorGP.jl" begin

    tt = MPT([rand(1,4,2),rand(2,5,3),rand(3,4,1)]);
    @test order(tt) == 3;

end

