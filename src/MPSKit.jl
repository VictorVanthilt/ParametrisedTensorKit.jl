const PTM = ParametrisedTensorMap

for type in [MPO, MPOHamiltonian]
    @eval function delay(H::$type{T}, dt::Number) where {TT,T<:SparseBlockTensorMap{TT}}
        return $type(map(parent(H)) do x
            return SparseBlockTensorMap{TT}(Dict(I => X isa PTM ? delay(X, dt) : X for (I, X) in nonzero_pairs(x)), codomain(x), domain(x))
        end)
    end

    @eval function (O::$type{T})(t) where {TT,T<:SparseBlockTensorMap{TT}}
        return $type(map(parent(O)) do x
            data = Dict(I => X isa PTM ? X(t) : X for (I, X) in nonzero_pairs(x))
            TT′ = valtype(data)
            return SparseBlockTensorMap{TT′}(data, codomain(x), domain(x))
        end)
    end
end