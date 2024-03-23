const TIMEDOP = ParametrisedTensorMap

function (H::MPOHamiltonian{T})(t) where {S,E<:Number,T<:BlockTensorMap{E,S,2,2}}
    return MPOHamiltonian(map(H.data) do x
        new_subtensors = Dict(I => old_subtensor isa TIMEDOP ? eval_coeff(old_subtensor, t) : old_subtensor for (I, old_subtensor) in nonzero_pairs(x))
        newx = BlockTensorMap{E,S,2,2}(undef, x.codom, x.dom)
        for (key, value) in new_subtensors
            newx[key] = value
        end
        return newx
    end)
end

function (H::SparseMPO{T})(t) where {E<:Number,S,T<:BlockTensorMap{E,S,2,2}}
    return InfiniteMPO(map(H.data) do x
        new_subtensors = Dict(I => old_subtensor isa TIMEDOP ? eval_coeff(old_subtensor, t) : old_subtensor for (I, old_subtensor) in nonzero_pairs(x))
        newx = BlockTensorMap{E,S,2,2}(undef, x.codom, x.dom)
        for (key, value) in new_subtensors
            newx[key] = value
        end
        return newx
    end)
end

MPSKit.ismpoidentity(::ParametrisedTensorMap) = false