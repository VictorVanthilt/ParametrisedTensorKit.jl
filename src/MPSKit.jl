const PTM = ParametrisedTensorMap

for type in [MPO]
    @eval function delay(H::$type{T}, dt::Number) where {TT,T<:SparseBlockTensorMap{TT}}
        return $type(map(parent(H)) do x
            return SparseBlockTensorMap{TT}(Dict(I => X isa PTM ? delay(X, dt) : X for (I, X) in nonzero_pairs(x)), space(x))
        end)
    end

    @eval function (O::$type{T})(t) where {TT,T<:SparseBlockTensorMap{TT}}
        return $type(map(parent(O)) do x
            data = Dict(I => X isa PTM ? X(t) : X for (I, X) in nonzero_pairs(x))
            TT′ = valtype(data)
            return SparseBlockTensorMap{TT′}(data, space(x))
        end)
    end
end

for type in [FiniteMPOHamiltonian, InfiniteMPOHamiltonian]
    @eval function (H::$type{T})(t) where {TT,T<:MPSKit.JordanMPOTensor{TT}}
        return $type(map(parent(H)) do x
            data = Dict(I => X isa PTM ? X(t) : X for (I, X) in nonzero_pairs(x))
            TT′ = valtype(data)
            sbtm = SparseBlockTensorMap{TT′}(data, space(x))
            return MPSKit.JordanMPOTensor(sbtm)
        end)
    end
end