module MPSTools

using MPSKit, MPSKitModels, TensorKit, ProgressBars

export state_to_mps, get_compbasis_vector, run_tdvp2, run_tempo, overlap

"""
Turn a comp. basisstate into an MPS

state: [(a, b), (c, d), ...]
"""
function state_to_mps(states)
    mps_states = Array{TrivialTensorMap}(undef, 0)

    for state in states
        state_tens = zeros(ComplexF64, 1, 2, 1)
        state_tens[1, 1, 1] = state[1]
        state_tens[1, 2, 1] = state[2]
        state_MPSTens = MPSTensor(state_tens)
        push!(mps_states, state_MPSTens)
    end

    # This is bs, but it works
    return FiniteMPS([mps_state for mps_state in mps_states], normalize=true, overwrite=true)
end

state_to_mps(state, N::Int) = state_to_mps([state for _ in 1:N])

"""
Get the computational basis vector from an MPS by interating over bitstrings
"""
function get_compbasis_vector(MPS::FiniteMPS)
    arr = convert(TensorMap, MPS)[]
    MPS_size = prod(size(arr))

    # Get amount of physical dimensions (there are 2 free virtual legs at the start and end)
    N = ndims(arr) - 2

    vector = zeros(ComplexF64, MPS_size)
    for i in 1:MPS_size
        # Get the bitstring corresponding to the index
        bs = bitstring(i-1)[end- N + 1: end]
        index = parse.(Int, split(bs, "")) # Convert to array of integers
        index .+= 1 # Julia is 1-indexed
        vector[i] = arr[1, index..., 1]
    end
    return vector/norm(vector)
end

function run_tdvp2(ψ₀, H, t0, dt, tend)
    ψ_tdvp = copy(ψ₀)
    # first step
    ψ_tdvp, envs = timestep(ψ₀, H, t0, dt, TDVP2())
    for t in ProgressBar((t0+dt):dt:tend)
        ψ_tdvp, envs = timestep!(ψ_tdvp, H, t, dt, TDVP2(), envs)
    end
    return ψ_tdvp
end

function run_tempo(ψ₀, O, t0, dt, tend)
    ψ_tempo = copy(ψ₀)
    for t in ProgressBar(t0:dt:tend)
        ψ_tempo, _, _ = approximate(ψ_tempo, (O(t), ψ_tempo), DMRG2(verbose=false))
    end
    return ψ_tempo
end

overlap(ψ₁::MPSKit.AbstractMPS, ψ₂::MPSKit.AbstractMPS) = abs(dot(ψ₁, ψ₂))

end