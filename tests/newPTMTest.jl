using Revise, BlockTensorKit, MPSKit, TensorKit, MPSKitModels

includet("../src/newParametrisedTensorKit.jl")
includet("../src/mpstools.jl")
using .ParametrisedTensorKit
using .MPSTools

f(t) = sin(t)
g(t) = cos(t)
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

Lat = FiniteChain(2)
H = @mpoham (f*σˣ() + PTMs + PTMs + B*B){Lat[1]} + (f*σˣ() + f*σʸ() + B){Lat[2]}
H(π/2)

ψ = state_to_mps([(1,0), (0,1)])
timestep(ψ, H, 1, 0.1, TDVP())[1]
timestep(ψ, H, 1, 0.1, TDVP2())[1]

f1 = t -> sin(t)
f2 = t -> cos(t)

H = @mpoham begin
    (f1*S_x() + PTMs){Lat[1]} + (f2*S_z()){Lat[2]}
end

# Multiple timesteps test

ψ = state_to_mps([(1,0), (0,1)])
ψ, envs = timestep(ψ, H, 1, 0.1, TDVP2())

for i in 1:10
    ψ, envs = timestep(ψ, H, 1, 0.1, TDVP2(), envs)
end