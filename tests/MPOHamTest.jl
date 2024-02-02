using Revise, BlockTensorKit, MPSKit, TensorKit, MPSKitModels
includet("../src/sumoftensors.jl")
includet("../src/parametrisedtensormap.jl")
includet("../src/mpstools.jl")

f = t -> sin(t)
fσx = f * S_x()

Lattice = FiniteChain(5)

H = @mpoham fσx{Lattice[1]} + fσx{Lattice[2]} + fσx{Lattice[1]}*fσx{Lattice[2]}

H(1)

H₋ = @mpoham fσx{Lattice[1]} - fσx{Lattice[2]} 

H₋(1)