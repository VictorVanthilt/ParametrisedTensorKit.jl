const TIMEDOP = ParametrisedTensorMap

# function (H::MPOHamiltonian{T})(t) where {S,T<:BlockTensorMap{S,2,2,<:Union{MPSKit.MPOTensor, ParametrisedTensorMap, TensorKit.BraidingTensor}}}
#     return MPOHamiltonian(map(H.data) do x
#         new_subtensors = Dict(I => old_subtensor isa TIMEDOP ? eval_coeff(old_subtensor, t) : old_subtensor for (I, old_subtensor) in nonzero_pairs(x))
#         new_tensortype = Union{(typeof.(values(new_subtensors)))...}

#         newx = BlockTensorMap{S,2,2,new_tensortype}(undef, x.codom, x.dom)
#         for (key, value) in new_subtensors
#             newx[key] = value
#         end
#         return newx
#     end)
# end

function (H::MPOHamiltonian{T})(t) where {S,E<:Number,T<:BlockTensorMap{S,2,2,E}}
    return MPOHamiltonian(map(H.data) do x
        new_subtensors = Dict(I => old_subtensor isa TIMEDOP ? eval_coeff(old_subtensor, t) : old_subtensor for (I, old_subtensor) in nonzero_pairs(x))
        newx = BlockTensorMap{S,2,2,E}(undef, x.codom, x.dom)
        for (key, value) in new_subtensors
            newx[key] = value
        end
        return newx
    end)
end