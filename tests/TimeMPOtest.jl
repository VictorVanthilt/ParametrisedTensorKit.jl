using Revise, BlockTensorKit, MPSKit, TensorKit, MPSKitModels

includet("../src/ParametrisedTensorKit.jl")
using .ParametrisedTensorKit

f(t) = sin(t)

data = Array{Any,3}(missing, 1, 3, 3)

data[1, 1, 1] = 1; # regular numbers are interpreted as identity operators
data[1, 1, 2] = -S_z();
data[1, 2, 3] = f*S_x();
data[1, 1, 3] = f*S_x();
data[1, 3, 3] = 1;

H = MPOHamiltonian(data)

time_mpo = make_time_mpo(H, 0.1, WI)

