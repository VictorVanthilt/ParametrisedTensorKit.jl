module MPSTools

using MPSKit, MPSKitModels, TensorKit

export state_to_mps, get_compbasis_vector

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

end