const PTM = ParametrisedTensorMap

function (H::MPOHamiltonian{T})(t) where {S,E<:Number,T<:BlockTensorMap{E,S,N₁,N₂}} where {N₁,N₂}
    return MPOHamiltonian(map(H.data) do x
        new_subtensors = Dict(I => old_subtensor isa PTM ? eval_coeff(old_subtensor, t) : old_subtensor for (I, old_subtensor) in nonzero_pairs(x))
        newx = BlockTensorMap{E,S,N₁,N₂}(undef, x.codom, x.dom)
        for (key, value) in new_subtensors
            newx[key] = value
        end
        return newx
    end)
end

function (H::SparseMPO{T})(t) where {E<:Number,S,T<:BlockTensorMap{E,S,N₁,N₂}} where {N₁,N₂}
    return InfiniteMPO(map(H.data) do x
        new_subtensors = Dict(I => old_subtensor isa PTM ? eval_coeff(old_subtensor, t) : old_subtensor for (I, old_subtensor) in nonzero_pairs(x))
        newx = BlockTensorMap{E,S,N₁,N₂}(undef, x.codom, x.dom)
        for (key, value) in new_subtensors
            newx[key] = value
        end
        return newx
    end)
end

# MPSKit.ismpoidentity(::ParametrisedTensorMap) = false

function delay(H::MPOHamiltonian{T}, dt::Number) where {E<:Number,S,T<:BlockTensorMap{E,S,N₁,N₂}} where {N₁,N₂}
    return MPOHamiltonian(map(H.data) do x
        new_subtensors = Dict(I => old_subtensor isa PTM ? delay(old_subtensor, dt) : old_subtensor for (I, old_subtensor) in nonzero_pairs(x))
        newx = BlockTensorMap{E,S,N₁,N₂}(undef, x.codom, x.dom)
        for (key, value) in new_subtensors
            newx[key] = value
        end
        return newx
    end)
end

function delay(H::SparseMPO{T}, dt::Number) where {E<:Number,S,T<:BlockTensorMap{E,S,N₁,N₂}} where {N₁,N₂}
    return InfiniteMPO(map(H.data) do x
        new_subtensors = Dict(I => old_subtensor isa PTM ? delay(old_subtensor, dt) : old_subtensor for (I, old_subtensor) in nonzero_pairs(x))
        newx = BlockTensorMap{E,S,N₁,N₂}(undef, x.codom, x.dom)
        for (key, value) in new_subtensors
            newx[key] = value
        end
        return newx
    end)
end