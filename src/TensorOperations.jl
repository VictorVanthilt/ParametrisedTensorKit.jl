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
    C = ParametrisedTensorMap(newTens, newCoeff)
    return C
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
    C = ParametrisedTensorMap(newTens, newCoeff)
    return C
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
    C = β * C + sum(ptms)
    return C
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

# It is crucial that we allocate a PTM of the right length, and then later overwrite the already allocated tensors and coeffs
# To prevent the inner ParametrisedTensorMap from discarding the allocated tensors due to tensors with 0-norms we add ones to the tensors
# This is not ideal, to make sure no accidents happen the coefficients are set to trivial functions that return 0
function TO.tensoralloc_contract(TC, A::ParametrisedTensorMap, pA::Index2Tuple, conjA::Bool,
                              B::AbstractTensorMap, pB::Index2Tuple, conjB::Bool,
                              pAB::Index2Tuple, istemp::Val=Val(false),
                              allocator=TO.DefaultAllocator())
    tensors = map(A.tensors) do a
        return TO.tensoralloc_contract(TC, a, pA, conjA, B, pB, conjB, pAB, istemp, allocator)
    end
    coeffs = Vector{Union{Number, Function}}(fill(NaN, length(A)))
    return ParametrisedTensorMap(tensors, coeffs)
end

function TO.tensoralloc_contract(TC, A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
                              B::ParametrisedTensorMap, pB::Index2Tuple, conjB::Bool,
                              pAB::Index2Tuple, istemp::Val=Val(false),
                              allocator=TO.DefaultAllocator())
    tensors = map(B.tensors) do b
        return TO.tensoralloc_contract(TC, A, pA, conjA, b, pB, conjB, pAB, istemp, allocator)
    end
    coeffs = Vector{Union{Number, Function}}(fill(NaN, length(B)))
    return ParametrisedTensorMap(tensors, coeffs)
end

function TO.tensoralloc_contract(TC, A::ParametrisedTensorMap, pA::Index2Tuple, conjA::Bool,
                              B::ParametrisedTensorMap, pB::Index2Tuple, conjB::Bool,
                              pAB::Index2Tuple, istemp::Val=Val(false),
                              allocator=TO.DefaultAllocator())
    tensors = map(Base.Iterators.product(A.tensors, B.tensors)) do (a, b)
        return TO.tensoralloc_contract(TC, a, pA, conjA, b, pB, conjB, pAB, istemp, allocator)
    end
    coeffs = Vector{Union{Number, Function}}(fill(NaN, length(A)*length(B)))
    return ParametrisedTensorMap(vec(tensors), coeffs)
end

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