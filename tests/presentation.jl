using Revise, MPSKit, TensorKit, MPSKitModels

includet("../src/ParametrisedTensorKit.jl")
using .ParametrisedTensorKit

f(t) = sin(t)
g(t) = cos(t)
T = σᶻ()

A = f*T
A(0)
A(π/2)

B = A*A
B(π/2)

C = T + A + B
C(π/2)

Lat = FiniteChain(2)

H = @mpoham (f*σᶻ()){Lat[1]} + (g*σˣ()){Lat[2]}

H(1)