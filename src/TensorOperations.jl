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

function TO.tensorcontract_type(TC, A::ParametrisedTensorMap, pA::Index2Tuple, conjA::Bool,
                                B::AbstractTensorMap, pB::Index2Tuple, conjB::Bool,
                                pAB::Index2Tuple)
    return ParametrisedTensorMap{TC}
end

function TO.tensorcontract_type(TC, A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
                                B::ParametrisedTensorMap, pB::Index2Tuple, conjB::Bool,
                                pAB::Index2Tuple)
    return ParametrisedTensorMap{TC}
end

# function TO.tensoralloc_contract(TC, A::ParametrisedTensorMap, pA::Index2Tuple, conjA::Bool,
#                               B::AbstractTensorMap, pB::Index2Tuple, conjB::Bool,
#                               pAB::Index2Tuple, istemp::Val=Val(false),
#                               allocator=TO.DefaultAllocator())
#                               @show "1111"
#     return ParametrisedTensorMap{TC}(tensoralloc_contract(TC, A.tensors[1], pA, conjA, B, pB, conjB, pAB, istemp, allocator))
# end

# function TO.tensoralloc_contract(TC, A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
#                               B::ParametrisedTensorMap, pB::Index2Tuple, conjB::Bool,
#                               pAB::Index2Tuple, istemp::Val=Val(false),
#                               allocator=TO.DefaultAllocator())
#         @show "2222"
#                               return ParametrisedTensorMap{TC}(tensoralloc_contract(TC, A, pA, conjA, B.tensors[1], pB, conjB, pAB, istemp, allocator))
# end

# function TO.tensoralloc_contract(TC, A::ParametrisedTensorMap, pA::Index2Tuple, conjA::Bool,
#                               B::ParametrisedTensorMap, pB::Index2Tuple, conjB::Bool,
#                               pAB::Index2Tuple, istemp::Val=Val(false),
#                               allocator=TO.DefaultAllocator())
#     tensor = tensoralloc_contract(TC, A.tensors[1], pA, conjA, B.tensors[1], pB, conjB, pAB, istemp, allocator)
#     @show "3333"
#     return ParametrisedTensorMap{TC}(tensor)
# end
function TO.tensorfree!(t::ParametrisedTensorMap, args...)
    for tensor in t.tensors
        tensorfree!(tensor, args...)
    end
    return nothing
end

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
    tensors = map(A.tensors) do t
        return tensortrace(pC, t, pA, conjA, 1)
    end
    coeffs = map(A.coeffs) do c
        return combinecoeff(α, c)
    end
    return ParametrisedTensorMap(tensors, coeffs)
end