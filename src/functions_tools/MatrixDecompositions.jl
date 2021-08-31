function eigen(mpo::MPO)
    F = eigen(mpo2mat(mpo));
    return F.values
end