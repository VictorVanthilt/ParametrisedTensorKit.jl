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
        newTens[i+1] = tensorcontract(pAB, A.tensors[i], pA, conjA, B, pB, conjB, 1)
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
        newTens[i+1] = tensorcontract(pAB, A, pA, conjA, B.tensors[i], pB, conjB, 1)
        newCoeff[i+1] = combinecoeff(α, B.coeffs[i])
    end
    return ParametrisedTensorMap(newTens, newCoeff)
end

function TO.tensorcontract!(C::ParametrisedTensorMap, pAB::Index2Tuple,
                            A::AbstractTensorMap, pA::Index2Tuple, conjA::Symbol,
                            B::AbstractTensorMap, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number)
    C *= β
    C += tensorcontract(pAB, A, pA, conjA, B, pB, conjB, α)

    return C
end

function TO.tensorcontract!(C::ParametrisedTensorMap, pAB::Index2Tuple,
                            A::ParametrisedTensorMap, pA::Index2Tuple, conjA::Symbol,
                            B::AbstractTensorMap, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number)
    C *= β
    C += α * tensorcontract(pAB, A, pA, conjA, B, pB, conjB, α)

    return C
end

function TO.tensorcontract!(C::ParametrisedTensorMap, pAB::Index2Tuple,
                            A::ParametrisedTensorMap, pA::Index2Tuple, conjA::Symbol,
                            B::ParametrisedTensorMap, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number)
    C *= β
    C += tensorcontract(pAB, A, pA, conjA, B, pB, conjB, α)

    return C
end

function TO.tensorcontract!(C::ParametrisedTensorMap, pAB::Index2Tuple,
                            A::AbstractTensorMap, pA::Index2Tuple, conjA::Symbol,
                            B::ParametrisedTensorMap, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number)
    C *= β
    C += tensorcontract(pAB, A, pA, conjA, B, pB, conjB, α)
    return C
end

# Distributivity
function TO.tensorcontract!(C::AbstractTensorMap, pAB::Index2Tuple,
                            A::ParametrisedTensorMap, pA::Index2Tuple, conjA::Symbol,
                            B::ParametrisedTensorMap, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number)
    ptms = Vector{ParametrisedTensorMap}(undef, length(A))

    for i in eachindex(ptms)
        ptms[i] = A.coeffs[i] * tensorcontract(pAB, A.tensors[i], pA, conjA, B, pB, conjB, α)

    end
    return β * C + sum(ptms)
end

TO.tensorfree!(t::ParametrisedTensorMap) = nothing

# tensortrace
function TO.tensortrace!(C::AbstractTensorMap, pC::Index2Tuple,
    A::ParametrisedTensorMap, pA::Index2Tuple, conjA::Symbol,
    α::Number, β::Number)
    C *= β
    C += tensortrace(pC, A, pA, conjA, α)
    return C
end

function TO.tensortrace(pC::Index2Tuple, A::ParametrisedTensorMap,
                        pA::Index2Tuple, conjA::Symbol, α::Number)

    tensors = Vector{typeof(tensortrace(pC, A.tensors[1], pA, conjA, 1))}(undef, length(A)) # this is horrible
    coeffs = Vector{Union{Number, Function}}(undef, length(A))
    for i in eachindex(A)
        tensors[i] = tensortrace(pC, A.tensors[i], pA, conjA, 1)
        coeffs[i] = combinecoeff(α, A.coeffs[i])
    end
    return ParametrisedTensorMap(tensors, coeffs)
end