using Revise, BlockTensorKit, MPSKit, TensorKit, MPSKitModels

includet("../src/ParametrisedTensorKit.jl")
includet("../src/mpstools.jl")
using .ParametrisedTensorKit
using .MPSTools

N = 5
lattice = FiniteChain(N)
f(t) = t
pulses = [f for _ in 1:(3N-1)]

function make_hamiltonian(pulsefuncs, lattice)
    N = length(lattice)
    triv = t -> 1
    H = @mpoham begin
        sum(1:N) do i
            return (pulsefuncs[i] * S_x()){lattice[i]}
        end +

        sum(1:N) do i
            j = N + i
            return (pulsefuncs[j] * S_y()){lattice[i]}
        end +

        sum(nearest_neighbours(lattice)) do (i, j)
            k = 2N + i.coordinates[1]
            return ((pulsefuncs[k]*σˣ()){i}*(triv*σˣ()){j} + (pulsefuncs[k]*σʸ()){i}*(triv*σʸ()){j})
        end
    end
    return H
end

H = make_hamiltonian(pulses, lattice)