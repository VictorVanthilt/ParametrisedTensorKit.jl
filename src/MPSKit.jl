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

function delay(H::MPOHamiltonian{T}, dt::Number) where {TT, T<:SparseBlockTensorMap{TT}}
    return MPOHamiltonian(map(parent(H)) do x
        return SparseBlockTensorMap{TT}(Dict(I => X isa PTM ? delay(X, dt) : X for (I, X) in nonzero_pairs(x)), codomain(x), domain(x))
    end)
end

function delay(O::MPO{T}, dt) where {TT, T<:SparseBlockTensorMap{TT}}
    return MPO(map(parent(O)) do x
        return SparseBlockTensorMap{TT}(Dict(I => X isa PTM ? delay(X, dt) : X for (I, X) in nonzero_pairs(x)), codomain(x), domain(x))
    end)
end