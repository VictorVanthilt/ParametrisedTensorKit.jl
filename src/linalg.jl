function LinearAlgebra.mul!(tC::TrivialTensorMap, tA::TrivialTensorMap, tB::ParametrisedTensorMap, α::Bool, β::Bool)
    coeffs = copy(tB.coeffs)
    tensors = map(tB.tensors) do t
        t′ = similar(tC)
        mul!(t′, tA, t, α, β)
        return t′
    end
    tC = ParametrisedTensorMap(tensors, coeffs)
    return tC
end

function LinearAlgebra.mul!(tC::TrivialTensorMap, tA::ParametrisedTensorMap, tB::TrivialTensorMap, α::Bool, β::Bool)
    coeffs = copy(tA.coeffs)
    tensors = map(tA.tensors) do t
        t′ = similar(tC)
        mul!(t′, t, tB, α, β)
        return t′
    end
    tC = ParametrisedTensorMap(tensors, coeffs)
    return tC
end

function LinearAlgebra.mul!(tC::TrivialTensorMap, tA::ParametrisedTensorMap, tB::ParametrisedTensorMap, α::Bool, β::Bool) # distributiviteit!
    newtensors = similar(Vector{typeof(tC)}, length(tA) * length(tB))
    newcoeffs = Vector{Union{Number,Function}}(undef, length(tA) * length(tB))
    for i in eachindex(tA)
        for j in eachindex(tB)
            index = (i - 1) * length(tB) + j
            newtensors[index] = tA.tensors[i] * tB.tensors[j]
            newcoeffs[index] = combinecoeff(tA.coeffs[i], tB.coeffs[j])
        end
    end
    tC = ParametrisedTensorMap(newtensors, newcoeffs)
    return tC
end