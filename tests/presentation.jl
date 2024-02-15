using Revise, MPSKit, TensorKit, MPSKitModels

includet("../src/ParametrisedTensorKit.jl")
using .ParametrisedTensorKit

f(t) = sin(t)

T = σᶻ()

fT = f*T

fT(0)
fT(3π/2)

f*fT
2*fT

fT*f

(f*fT*fT*f)(π/2)

SOT = f*T + T + 2*(f*T)