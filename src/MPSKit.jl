const PTM = ParametrisedTensorMap

function (H::MPOHamiltonian{T})(t) where {TT, T<:SparseBlockTensorMap{TT}}
    return MPOHamiltonian(map(parent(H)) do x
        return SparseBlockTensorMap{TT}(Dict(I => X isa PTM ? X(t) : X for (I, X) in nonzero_pairs(x)), codomain(x), domain(x))
    end)
end

function (O::MPO{T})(t) where {TT, T<:SparseBlockTensorMap{TT}}
    return MPO(map(parent(O)) do x
        return SparseBlockTensorMap{TT}(Dict(I => X isa PTM ? X(t) : X for (I, X) in nonzero_pairs(x)), codomain(x), domain(x))
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