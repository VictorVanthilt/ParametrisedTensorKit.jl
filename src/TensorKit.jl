# TensorKit
# ==========
# Ultra Mega Janky
TK.isometry(::Type{T}, args...) where {T<:AbstractTensorMap} = isometry(Matrix{ComplexF64}, args...)

TK.domain(t::ParametrisedTensorMap) = domain(t.tensors[1])
TK.codomain(t::ParametrisedTensorMap) = codomain(t.tensors[1])

TK.storagetype(::Type{<:ParametrisedTensorMap{E,S,N1,N2,T}}) where {E,S,N1,N2,T} = TensorKit.storagetype(T)

TK.has_shared_permute(t::ParametrisedTensorMap, args...) = false

function TK.add_transform!(tdst::TensorMap{T,S,N₁,N₂},
    tsrc::ParametrisedTensorMap,
    (p₁, p₂)::Index2Tuple{N₁,N₂},
    fusiontreetransform,
    α::Number,
    β::Number,
    backend::AbstractBackend...) where {T,S,N₁,N₂}
    tdst = ParametrisedTensorMap(zerovector(tdst)) # convert the destination to a PTM
    return TK.add_transform!(tdst, tsrc, (p₁, p₂), fusiontreetransform, α, β, backend...)
end

function TK._add_trivial_kernel!(tdst::ParametrisedTensorMap, tsrc::ParametrisedTensorMap, (p₁, p₂), fusiontreetransform, α, β, backend...)
    tensors = map(tsrc.tensors) do t
        t′ = similar(tdst.tensors[1])
        TK._add_trivial_kernel!(t′, t, (p₁, p₂), fusiontreetransform, α, β, backend...)
        return t′
    end
    coeffs = deepcopy(tsrc.coeffs)
    tdst = ParametrisedTensorMap(tensors, coeffs)
    return nothing
end

TK.space(t::ParametrisedTensorMap) = space(t.tensors[1]) 

# TK.storagetype(t::AbstractTensorMap) = Matrix{scalartype(t)}