function LinearAlgebra.mul!(tC::TensorMap, tA::TensorMap, tB::ParametrisedTensorMap, α, β)
    coeffs = copy(tB.coeffs)
    tensors = map(tB.tensors) do t
        t′ = similar(tC)
        mul!(t′, tA, t, α, β)
        return t′
    end
    tC = ParametrisedTensorMap(tensors, coeffs)
    return tC
end

function LinearAlgebra.mul!(tC::TensorMap, tA::ParametrisedTensorMap, tB::TensorMap, α, β)
    coeffs = copy(tA.coeffs)
    tensors = map(tA.tensors) do t
        t′ = similar(tC)
        mul!(t′, t, tB, α, β)
        return t′
    end
    tC = ParametrisedTensorMap(tensors, coeffs)
    return tC
end

function LinearAlgebra.mul!(tC::TensorMap, tA::ParametrisedTensorMap, tB::ParametrisedTensorMap, α, β)
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

function LinearAlgebra.mul!(tC::ParametrisedTensorMap, tA::ParametrisedTensorMap, tB::ParametrisedTensorMap, α, β)
    newtensors = similar(Vector{typeof(tC)}, length(tA) * length(tB))
    newcoeffs = Vector{Union{Number,Function}}(undef, length(tA) * length(tB))
    for i in eachindex(tA)
        for j in eachindex(tB)
            index = (i - 1) * length(tB) + j
            newtensors[index] = tA.tensors[i] * tB.tensors[j]
            newcoeffs[index] = combinecoeff(combinecoeff(tA.coeffs[i], tB.coeffs[j]), α)
        end
    end
    tC = β*tC + ParametrisedTensorMap(newtensors, newcoeffs)
    return tC
end

function LinearAlgebra.mul!(tC::ParametrisedTensorMap, tA::ParametrisedTensorMap, tB::AbstractTensorMap, α, β)
    return tC = β*tC + α * tA * tB
end

# very poor definition, supposed to only give an indication!
function LinearAlgebra.norm(t::ParametrisedTensorMap)
    @warn "Calculating a norm of a ParametrisedTensorMap is not well defined and may not be useful" maxlog=1
    nm = 0
    for i in eachindex(t)
        if t.coeffs[i] isa Number
            nm += norm(t.coeffs[i]) * norm(t.tensors[i])
        else
            nm += norm(t.tensors[i])
        end
    end
    return nm
end