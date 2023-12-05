using BlockTensorKit, MPSKit, TensorKit
include("../src/parametrisedtensormap.jl")

function coeff(t)
    return t<1 ? t : 1 
end

sz = [ 1 0
      0 -1]

sz = TensorMap(matrix, ℝ^2, ℝ^2)

PTM = ParametrisedTensorMap(t, coeff)

MPOHamiltonian(PTM)

function MPSKit.MPOHamiltonian(PTM::ParametrisedTensorMap)
    return MPO(PTM.tensor)
end