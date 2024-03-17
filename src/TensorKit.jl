# TensorKit
# ==========
# Ultra Mega Janky
TensorKit.isometry(::Type{T}, args...) where {T<:AbstractTensorMap} = isometry(Matrix{ComplexF64}, args...)
