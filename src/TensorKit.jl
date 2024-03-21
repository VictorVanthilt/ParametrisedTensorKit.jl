# TensorKit
# ==========
# Ultra Mega Janky
TensorKit.isometry(::Type{T}, args...) where {T<:AbstractTensorMap} = isometry(Matrix{ComplexF64}, args...)

TensorKit.domain(t::ParametrisedTensorMap) = domain(t.tensors[1])
TensorKit.codomain(t::ParametrisedTensorMap) = codomain(t.tensors[1])

TensorKit.storagetype(::Type{<:ParametrisedTensorMap{E,S,N1,N2,T}}) where {E,S,N1,N2,T} = TensorKit.storagetype(T)

TensorKit.has_shared_permute(t::ParametrisedTensorMap, args...) = false

function TensorKit.similar(t::ParametrisedTensorMap, T::Type, P::TensorMapSpace)
    tens = similar(t.tensors, T, P)
    return ParametrisedTensorMap(tens)
end
