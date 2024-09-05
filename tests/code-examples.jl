using MPSKit, MPSKitModels, TensorKit, BlockTensorKit, VectorInterface
include("../src/ParametrisedTensorKit.jl")
using .ParametrisedTensorKit

# Creating a ptm

f(t) =  sin(t);
g(t) = cos(t);

fσˣ = f*σˣ()
gσᶻ = g*σᶻ()

fσˣ(0)
gσᶻ(0)

σᶻ() + fσˣ + gσᶻ
0*σᶻ() + fσˣ + gσᶻ

# adding a bunch of constant tensors
σᶻ() + fσˣ + σᶻ() + gσᶻ + σᶻ() + fσˣ + gσᶻ + 0 * σʸ() + fσˣ(0)


# multiplying two ptms

2 * fσˣ
(fσˣ + gσᶻ) * (fσˣ + gσᶻ)

# adding ptms

ptm = σᶻ() + fσˣ + gσᶻ + 0 * σʸ() + fσˣ(0)
ptm(π)

zer = ParametrisedTensorMap(σᶻ(), 0)

Λ = FiniteChain(2)
H = @mpoham ((σˣ() + gσᶻ){Λ[1]} + gσᶻ{Λ[2]})

N = 4
Λ = FiniteChain(N)
H = @mpoham begin
    sum(1:N) do N
    return fσˣ{Λ[N]}
    end +
    sum(1:N) do n
    return gσᶻ{Λ[n]}
    end + 
    sum(1:N-1) do N
    return fσˣ{Λ[N]} * ((t->1) *  σˣ()){Λ[N+1]}
    end
end

# benchmark
using BenchmarkTools
b = @benchmark H(π/4);

H = @mpoham fσˣ{Λ[1]} + gσᶻ{Λ[2]}

H(π/4)
dt = 0.1

t0, dt, t1 = 0, 0.01, 40
make_time_mpo(H, dt, TaylorCluster(2, false, false))

for i in 1:100
    H^3
    println(i)
end