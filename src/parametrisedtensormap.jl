using MPSKit, TensorKit

struct ParametrisedTensorMap{S,N1,N2,T<:AbstractTensorMap{S,N1,N2},E} <: AbstractTensorMap{S,N1,N2}
    tensor::T
    coeff::E
end

function ParametrisedTensorMap(tensor::T, coeff::E) where {S,N1,N2,T<:AbstractTensorMap{S,N1,N2},E}
    return ParametrisedTensorMap{S,N1,N2,T,E}(tensor, coeff)
end

TensorKit.domain(t::ParametrisedTensorMap) = domain(t.tensor)
TensorKit.codomain(t::ParametrisedTensorMap) = codomain(t.tensor)

function (PTM::ParametrisedTensorMap)(t::Number)
    return PTM.coeff(t) * PTM.tensor
end

TensorKit.storagetype(::Type{<:ParametrisedTensorMap{S,N1,N2,T}}) where {S,N1,N2,T} = TensorKit.storagetype(T)

TensorKit.block(t::Type{<:ParametrisedTensorMap{S,N1,N2,T}}, triv::Trivial) where {S,N1,N2,T} = TensorKit.block(t.tensor.data, triv)

