using BlockTensorKit, MPSKit, TensorKit, MPSKitModels
include("../src/parametrisedtensormap.jl")

# Sinusoidal coefficient
coeff(t) = sin(t*2π)
sinSx = ParametrisedTensorMap(S_x(), coeff)

# Jordan block mpo form
# 1 C D
# 0 A B
# 0 0 1

T = ComplexF64
X = TensorMap(T[0 1; 1 0], ℂ^2 ← ℂ^2)
Z = TensorMap(T[1 0; 0 -1], ℂ^2 ← ℂ^2)

data = Array{Any,3}(missing, 1, 3, 3)
data[1, 1, 1] = identity(ℂ^2)
data[1, 1, 1] = one(T) # regular numbers are interpreted as identity operators
data[1, 1, 2] = -Z
data[1, 2, 3] = Z
data[1, 1, 3] = 3 * X
H_Ising = MPOHamiltonian(data);


# Ising Hamiltonian mpo
# 1 -Z h(t)X
# 0  0  Z
# 0  0  1


T = ComplexF64
Z = TensorMap(T[1 0; 0 -1], ℂ^2 ← ℂ^2)
hX = sinSx

data = Array{Any,3}(missing, 1, 3, 3)

data[1, 1, 1] = one(T) # regular numbers are interpreted as identity operators
data[1, 1, 2] = -Z
data[1, 2, 3] = Z
data[1, 1, 3] = hX
data[1, 3, 3] = one(T)

H_Ising = MPOHamiltonian(data);