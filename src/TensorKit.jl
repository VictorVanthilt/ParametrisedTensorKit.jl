# TensorKit
# ==========
# Ultra Mega Janky
TK.isometry(::Type{T}, args...) where {T<:AbstractTensorMap} = isometry(Matrix{ComplexF64}, args...)

TK.domain(t::ParametrisedTensorMap) = domain(t.tensors[1])
TK.codomain(t::ParametrisedTensorMap) = codomain(t.tensors[1])

TK.storagetype(::Type{<:ParametrisedTensorMap{E,S,N1,N2,T}}) where {E,S,N1,N2,T} = TensorKit.storagetype(T)

TK.has_shared_permute(t::ParametrisedTensorMap, ::TK.Index2Tuple) = false

function TK.add_transform!(tdst::TensorMap{T,S,N₁,N₂},
    tsrc::ParametrisedTensorMap,
    (p₁, p₂)::Index2Tuple{N₁,N₂},
    fusiontreetransform,
    α::Number,
    β::Number,
    backend::AbstractBackend...) where {T,S,N₁,N₂}
    # @assert tdst isa ParametrisedTensorMap "The destination tensor must be a ParametrisedTensorMap of length(tsrc)"
    tensors = map(tsrc.tensors) do t
        t′ = similar(tdst)
        TK.add_transform!(t′, t, (p₁, p₂), fusiontreetransform, α, β, backend...)
        return t′
    end
    return ParametrisedTensorMap(tensors, deepcopy(tsrc.coeffs))
end

function TK.add_transform!(tdst::ParametrisedTensorMap{E,S,N₁,N₂},
    tsrc::ParametrisedTensorMap,
    (p₁, p₂)::Index2Tuple{N₁,N₂},
    fusiontreetransform,
    α::Number,
    β::Number,
    backend::AbstractBackend...) where {E,S,N₁,N₂}

    # TODO: sometimes there's too many tensors in the destination; where is the destination allocated?
    @assert length(tdst) >= length(tsrc) "The number of tensors in the destination and source must be the same"

    for i in eachindex(tsrc)
        tdst.coeffs[i] = deepcopy(tsrc.coeffs[i])
        TK.add_transform!(tdst.tensors[i], tsrc.tensors[i], (p₁, p₂), fusiontreetransform, α, β, backend...)
    end
    return tdst
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

function TK.:⊗(t1::ParametrisedTensorMap, t2::AbstractTensorMap)
    newtensors = map(t1.tensors) do t
        TK.:⊗(t, t2)
    end
    return ParametrisedTensorMap(newtensors, deepcopy(t1.coeffs))
end

function TK.:⊗(t1::AbstractTensorMap, t2::ParametrisedTensorMap)
    newtensors = map(t2.tensors) do t
        TK.:⊗(t1, t)
    end
    return ParametrisedTensorMap(newtensors, deepcopy(t2.coeffs))
end

function TK.:⊗(t1::ParametrisedTensorMap, t2::ParametrisedTensorMap)
    ptms = map(eachindex(t2)) do i
        TK.:⊗(t1, t2.tensors[i]) * t2.coeffs[i]
    end
    return sum(ptms)
end

# TK.storagetype(t::AbstractTensorMap) = Matrix{scalartype(t)}