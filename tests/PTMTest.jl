using Revise, BlockTensorKit, MPSKit, TensorKit, MPSKitModels

includet("../src/ParametrisedTensorKit.jl")
using .ParametrisedTensorKit

# Sinusoidal coefficient
coeff(t) = sin(t*2π)
typeof(coeff)
sinSx = ParametrisedTensorMap(S_x(), coeff)

# Ising Hamiltonian mpo
# 1 -Z h(t)X
# 0  0  Z
# 0  0  1


T = ComplexF64
Z = TensorMap(T[1 0; 0 -1], ℂ^2 ← ℂ^2)
hX = sinSx

data = Array{Any,3}(missing, 1, 3, 3)

data[1, 1, 1] = one(T); # regular numbers are interpreted as identity operators
data[1, 1, 2] = -Z;
data[1, 2, 3] = hX;
data[1, 1, 3] = hX;
data[1, 3, 3] = one(T);

H_Ising = MPOHamiltonian(data)

O_Ising = make_time_mpo(H_Ising, 0.1, TaylorCluster()) #TODO

ψ₀ = FiniteMPS(rand, ComplexF64, 5, ℂ^2, ℂ^2)

ψ₁, _ = timestep(ψ₀, H_Ising, 1, .01, TDVP());

ψ₂, _ = timestep(ψ₀, H_Ising, 1, .01, TDVP2());

# Batch TDVP test
ψ, env = timestep(ψ₀, H_Ising,  .25, .25, TDVP2());

for i in 1:100
    ψ, env = timestep(ψ, H_Ising,  .25, .25, TDVP(), env);
end

# Batch TDVP2 test
ψ, env = timestep(ψ₀, H_Ising,  .25, .25, TDVP2());

for i in 1:100
    ψ, env = timestep(ψ, H_Ising,  .25, .25, TDVP2(), env);
end

# ===================
# Sum of tensors test
# ===================

f = t -> sin(t)
fX = ParametrisedTensorMap(S_x(), f)
fZ = ParametrisedTensorMap(S_z(), f)

sumfXZ = fX + fZ

data_sum = Array{Any,3}(missing, 1, 3, 3)

data_sum[1, 1, 1] = 1;
data_sum[1, 1, 2] = fX;
data_sum[1, 1, 3] = sumfXZ;
data_sum[1, 2, 3] = fZ;
data_sum[1, 3, 3] = 1;

H = MPOHamiltonian(data_sum)

ψ₀ = FiniteMPS(rand, ComplexF64, 10, ℂ^2, ℂ^2)

ψ₁, _ = timestep(ψ₀, H,  1, .01, TDVP());

ψ₂, _ = timestep(ψ₀, H,  1, .01, TDVP2());

# Batch TDVP test
ψ, env = timestep(ψ₀, H,  .25, .25, TDVP2());

for i in 1:100
    ψ, env = timestep(ψ, H,  .25, .25, TDVP(), env);
end

# Batch TDVP2 test
ψ, env = timestep(ψ₀, H,  .25, .25, TDVP2());

for i in 1:100
    ψ, env = timestep(ψ, H,  .25, .25, TDVP2(), env);
end
