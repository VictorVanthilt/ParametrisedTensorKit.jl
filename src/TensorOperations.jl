# TensorOperations
# ----------------

function TO.tensorcontract!(C::AbstractTensorMap, pAB::Index2Tuple,
                            A::ParametrisedTensorMap, pA::Index2Tuple, conjA::Symbol,
                            B::AbstractTensorMap, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number)

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

function TO.tensorcontract!(C::AbstractTensorMap, pAB::Index2Tuple,
                            A::AbstractTensorMap, pA::Index2Tuple, conjA::Symbol,
                            B::ParametrisedTensorMap, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number)

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

# Distributivity
function TO.tensorcontract!(C::AbstractTensorMap, pAB::Index2Tuple,
                            A::ParametrisedTensorMap, pA::Index2Tuple, conjA::Symbol,
                            B::ParametrisedTensorMap, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number)
    C_copy = copy(C)
    ptms = Vector{ParametrisedTensorMap}(undef, length(A))

    for i in eachindex(ptms)
        ptms[i] = A.coeffs[i] * tensorcontract!(C, pAB, A.tensors[i], pA, conjA, B, pB, conjB, α, 0)
    end
    return β * C_copy + sum(ptms)
end

TO.tensorfree!(t::ParametrisedTensorMap) = nothing