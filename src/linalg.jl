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
