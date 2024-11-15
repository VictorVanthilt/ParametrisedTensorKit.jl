# VectorInterface
# ---------------

function VI.scale!(ty::AbstractTensorMap, tx::ParametrisedTensorMap, α::Number)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))

    newTens   = copy(tx.tensors)
    newCoeffs = similar(tx.coeffs,  length(tx))
    
    for i in eachindex(tx.coeffs)
        newCoeffs[i] = combinecoeff(tx.coeffs[i], α)
    end
    ty = ParametrisedTensorMap(newTens, newCoeffs)
    return ty
end

function VI.scale!(t::ParametrisedTensorMap, α::Number)
    for i in eachindex(t)
        t.coeffs[i] = combinecoeff(t.coeffs[i], α)
    end
end

function VI.add!(ty::AbstractTensorMap, tx::ParametrisedTensorMap, α::Number, β::Number)
    ty = scale(ty, β) + scale(tx, α)
    return ty
end

function VI.zerovector(t::ParametrisedTensorMap)
    return ParametrisedTensorMap(zerovector(t.tensors[1]), 1)
end

function VI.zerovector!(t::ParametrisedTensorMap)
    for i in eachindex(t.tensors)
        zerovector!(t.tensors[i])
        t.coeffs[i] = 0
    end
    return t
end

function VI.add!(ty::ParametrisedTensorMap, tx::AbstractTensorMap, α::Number, β::Number)
    scale!(ty, β)
    ty += scale(tx, α)
    return ty
end

function VI.add!(ty::ParametrisedTensorMap, tx::ParametrisedTensorMap, α::Number, β::Number)
    scale!(ty, β)
    ty += scale(tx, α)
    return ty
end

LinearAlgebra.norm(::VectorInterface.Zero) = VectorInterface.Zero()

VI.scalartype(t::ParametrisedTensorMap{E}) where E = E
VI.scalartype(TT::Type{<:ParametrisedTensorMap{E}}) where E = E