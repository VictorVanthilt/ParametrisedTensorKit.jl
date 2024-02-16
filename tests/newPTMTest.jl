using Revise, BlockTensorKit, MPSKit, TensorKit, MPSKitModels

includet("../src/newParametrisedTensorKit.jl")
using .ParametrisedTensorKit

f(t) = sin(t)
T = S_x()

PTM = ParametrisedTensorMap([T,T], [f,f])