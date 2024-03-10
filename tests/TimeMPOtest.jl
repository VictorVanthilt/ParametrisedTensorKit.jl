using Revise, BlockTensorKit, MPSKit, TensorKit, MPSKitModels

includet("../src/ParametrisedTensorKit.jl")
using .ParametrisedTensorKit

f(t) = sin(t)
g(t) = cos(t)

N = 1
Lat = FiniteChain(N)

H = @mpoham (f*σᶻ()){Lat[1]} + (g*σˣ()){Lat[1]}

time_mpo = make_time_mpo(H, 0.1, WI)

