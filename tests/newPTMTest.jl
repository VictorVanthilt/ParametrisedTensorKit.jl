using Revise, BlockTensorKit, MPSKit, TensorKit, MPSKitModels

includet("../src/newParametrisedTensorKit.jl")
using .ParametrisedTensorKit

f(t) = sin(t)
T = S_x()

PTM = ParametrisedTensorMap(T, f)
PTMs = ParametrisedTensorMap([T,T,T], [f,2,f])
PTMs = ParametrisedTensorMap([T,T,T], Vector{Union{Number, Function}}([f,2,f]))

domain(PTMs)
codomain(PTMs)
storagetype(PTMs)

PTM(1)
PTMs(1.0)

T + PTM
T + PTMs
PTM + T
PTMs + T
PTM + PTMs
PTMs + PTM
PTM + PTM
PTMs + PTMs

length(PTMs + PTM)

A = PTM * PTMs
B = PTMs * PTMs