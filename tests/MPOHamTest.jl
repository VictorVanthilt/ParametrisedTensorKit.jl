using Revise, BlockTensorKit, MPSKit, TensorKit, MPSKitModels

includet("../src/ParametrisedTensorKit.jl")
using .ParametrisedTensorKit

function f(t)
    return sin(t)
end

function g(t)
    return cos(t)
end

fσx = f * S_x()
fσz = f * S_z()

gσx = g * S_x()
gσz = g * S_z()

sot1 = fσx + fσz
sot2 = gσx + gσz

Lattice = FiniteChain(2)

H = @mpoham fσx{Lattice[1]} + gσz{Lattice[2]}

H(1)

H_sum = @mpoham sot1{Lattice[1]} + sot2{Lattice[2]}

H_sum(1)

H_mixed = @mpoham fσx{Lattice[1]} + sot1{Lattice[2]}

LL = FiniteChain(100000) #Large Lattice
H_large = @mpoham sum(fσx{LL[i]} for i in 1:length(LL));