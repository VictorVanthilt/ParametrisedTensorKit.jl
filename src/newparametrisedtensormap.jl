import TensorOperations.tensorcontract!
import LinearAlgebra.mul!
import LinearAlgebra.lmul!

struct ParametrisedTensorMap{S,N1,N2,T<:AbstractTensorMap{S,N1,N2}} <: AbstractTensorMap{S,N1,N2}
    tensor::Vector{T}
    coeffs::Vector{Union{Number, Function}}
end

function ParametrisedTensorMap(tensor::T, coeff) where {S,N1,N2,T<:AbstractTensorMap{S,N1,N2}}
    return ParametrisedTensorMap{S,N1,N2,T}([tensor], [coeff])
end

function ParametrisedTensorMap(tensors::Vector{T}, coeffs::Vector{Union{Number, Function}}) where {S,N1,N2,T<:AbstractTensorMap{S,N1,N2}}
    return ParametrisedTensorMap{S,N1,N2,T}(tensors, coeffs)
end