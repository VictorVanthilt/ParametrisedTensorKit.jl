# TensorKit
# ==========
# Ultra Mega Janky
TK.isometry(::Type{T}, args...) where {T<:AbstractTensorMap} = isometry(Matrix{ComplexF64}, args...)

TK.domain(t::ParametrisedTensorMap) = domain(t.tensors[1])
TK.codomain(t::ParametrisedTensorMap) = codomain(t.tensors[1])

TK.storagetype(::Type{<:ParametrisedTensorMap{E,S,N1,N2,T}}) where {E,S,N1,N2,T} = TensorKit.storagetype(T)

TK.has_shared_permute(t::ParametrisedTensorMap, args...) = false

# TK.storagetype(t::AbstractTensorMap) = Matrix{scalartype(t)}