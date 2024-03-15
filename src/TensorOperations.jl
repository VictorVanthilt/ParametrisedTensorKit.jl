# TensorOperations
# ----------------

function TO.tensorcontract!(C::AbstractTensorMap{S}, pAB::Index2Tuple,
                            A::ParametrisedTensorMap, pA::Index2Tuple, conjA::Symbol,
                            B::AbstractTensorMap{S}, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number) where {S}

    newTens = Vector{typeof(C)}(undef, length(A) + 1)
    newCoeff = similar(A.coeffs, length(A) + 1)

    newTens[1] = C
    newCoeff[1] = β

    for i in eachindex(A.tensors)
        newTens[i+1] = tensorcontract!(C, pAB, A.tensors[i], pA, conjA, B, pB, conjB, 1, 0)
        newCoeff[i+1] = combinecoeff(α, A.coeffs[i])
    end
    return ParametrisedTensorMap(newTens, newCoeff)
end

function TO.tensorcontract!(C::AbstractTensorMap{S,N₁,N₂}, pAB::Index2Tuple,
                            A::AbstractTensorMap{S}, pA::Index2Tuple, conjA::Symbol,
                            B::ParametrisedTensorMap{S,N₁,N₂,T}, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number) where {S,N₁,N₂,T}

    newTens = Vector{typeof(C)}(undef, length(B) + 1)
    newCoeff = similar(B.coeffs, length(B) + 1)

    newTens[1] = C
    newCoeff[1] = β

    for i in eachindex(B.tensors)
        newTens[i+1] = tensorcontract!(C, pAB, A, pA, conjA, B.tensors[i], pB, conjB, 1, 0)
        newCoeff[i+1] = combinecoeff(α, B.coeffs[i])
    end
    return ParametrisedTensorMap(newTens, newCoeff)
end

TO.tensorfree!(t::ParametrisedTensorMap{S,N₁,N₂,T}) where {S,N₁,N₂,T}  = nothing