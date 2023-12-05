using MPSKit, TensorKit

# struct ParametrisedTensorMap{S, N1, N2, T<:AbstractTensorMap{S, N1, N2}} <: AbstractTensorMap{S, N1, N2}
#     tensor::T
#     dom::ProductSpace{S, N1}
#     codom::ProductSpace{S, N2}
#     coeff

#     function ParametrisedTensorMap(tensor::T, coeff) where T<:AbstractTensorMap
#         dom = tensor.dom
#         codom = tensor.codom

#         S = space(tensor, 1)
#         N1 = dim(dom)
#         N2 = dim(codom)
#         new{S, N1, N2, T}(tensor, dom, codom, coeff)
#     end

# end

# function Base.*(t1::ParametrisedTensorMap, t2::ParametrisedTensorMap)
#     return ParametrisedTensorMap(t1.tensor * t2.tensor, t1.coeff * t2.coeff)
# end

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