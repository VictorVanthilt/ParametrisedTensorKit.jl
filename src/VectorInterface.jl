# VectorInterface
# ---------------

function VI.scale!(ty::AbstractTensorMap, tx::ParametrisedTensorMap, α::Number)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))

    newTens   = copy(tx.tensors)
    newCoeffs = similar(tx.coeffs,  length(tx))
    
    for i in eachindex(tx.coeffs)
        newCoeffs[i] = combinecoeff(tx.coeffs[i], α)
    end
    return ParametrisedTensorMap(newTens, newCoeffs)
end

VI.scale!(t::ParametrisedTensorMap, α::Number) = α * t

VI.add!(ty::AbstractTensorMap, tx::ParametrisedTensorMap, α::Number, β::Number) = scale!(ty, β) + scale!(tx, α)

function VI.zerovector(t::ParametrisedTensorMap)
    return ParametrisedTensorMap(zero(t.tensors[1]), 1)
end

function VI.zerovector!(t::ParametrisedTensorMap)
    for i in eachindex(t.tensors)
        t.tensors[i] = zero(t.tensors[i])
        t.coeffs[i] = 0
    end
    return t
end

function VI.add!(ty::ParametrisedTensorMap, tx::AbstractTensorMap, α::Number)
    ty += scale!(tx, α)
end

function VI.add!(ty::ParametrisedTensorMap, tx::ParametrisedTensorMap, α::Number)
    ty += scale!(tx, α)
end