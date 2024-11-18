# VectorInterface
# ---------------

# scalartype
# ----------
VI.scalartype(t::ParametrisedTensorMap{E}) where {E} = E
VI.scalartype(TT::Type{<:ParametrisedTensorMap{E}}) where {E} = E

# zerovector
function VI.zerovector(t::ParametrisedTensorMap)
    return ParametrisedTensorMap(zerovector(t.tensors[1]), 1)
end

function VI.zerovector(t::ParametrisedTensorMap, S::Type{<:Number})
    return ParametrisedTensorMap(zerovector(t.tensors[1], S), S(VI.One()))
end

function VI.zerovector!(t::ParametrisedTensorMap)
    for i in eachindex(t.tensors)
        zerovector!(t.tensors[i])
        t.coeffs[i] = 0
    end
    return t
end

function VI.zerovector!!(t::ParametrisedTensorMap)
    return zerovector!(t)
end

# zerovector!!(t::ParametrisedTensorMap, S::Type{<:Number}) implemented in TensorKit

# scale
# -----
# scale(t::ParametrisedTensorMap, α::Number) implemented in TensorKit

function VI.scale!(t::ParametrisedTensorMap, α::Number)
    for i in eachindex(t)
        t.coeffs[i] = combinecoeff(t.coeffs[i], α)
    end
end

# scale!!(t::ParametrisedTensorMap, α::Number) implemented in TensorKit

function VI.scale!(ty::AbstractTensorMap, tx::ParametrisedTensorMap, α::Number)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    ty isa ParametrisedTensorMap || throw(ArgumentError("ty must be a ParametrisedTensorMap"))

    L_tx = length(tx)

    # resize ty to have length L_tx
    resize!(ty.tensors, L_tx)
    resize!(ty.coeffs, L_tx)

    for i in 1:L_tx
        ty.tensors[i] = tx.tensors[i]
        ty.coeffs[i] = combinecoeff(tx.coeffs[i], α)
    end
    return ty
end

# scale!!(ty::AbstractTensorMap, tx::ParametrisedTensorMap, α::Number) implemented in TensorKit

# add
# ---
# add(a, b) implemented in TensorKit
function VI.add(ty::ParametrisedTensorMap, tx::AbstractTensorMap, α::Number, β::Number)
    tz = copy(ty)
    scale!(tz, β)
    push!(tz.tensors, tx)
    push!(tz.coeffs, α)
    return tz
end

function VI.add(ty::AbstractTensorMap, tx::ParametrisedTensorMap, α::Number, β::Number)
    tz = copy(tx)
    scale!(tz, α)
    push!(tz.tensors, ty)
    push!(tz.coeffs, β)
    return tz
end

function VI.add(ty::ParametrisedTensorMap, tx::ParametrisedTensorMap, α::Number, β::Number)
    tz = copy(ty)
    scale!(tz, β)
    for i in eachindex(tx)
        push!(tz.tensors, tx.tensors[i])
        push!(tz.coeffs, combinecoeff(tx.coeffs[i], α))
    end
    return tz
end

function VI.add!(ty::AbstractTensorMap, tx::ParametrisedTensorMap, α::Number, β::Number)
    @assert ty isa ParametrisedTensorMap "cannot in-place add a ParametrisedTensorMap to a non-ParametrisedTensorMap"
    scale!(ty, β)
    for i in eachindex(tx)
        push!(ty.tensors, tx.tensors[i])
        push!(ty.coeffs, combinecoeff(tx.coeffs[i], α))
    end
    return ty
end

function VI.add!(ty::ParametrisedTensorMap, tx::AbstractTensorMap, α::Number, β::Number)
    scale!(ty, β)
    push!(ty.tensors, tx)
    push!(ty.coeffs, α)
    return ty
end

function VI.add!(ty::ParametrisedTensorMap, tx::ParametrisedTensorMap, α::Number, β::Number)
    scale!(ty, β)
    for i in eachindex(tx)
        push!(ty.tensors, tx.tensors[i])
        push!(ty.coeffs, combinecoeff(tx.coeffs[i], α))
    end
    return ty
end

function VI.add!!(tx::AbstractTensorMap, ty::ParametrisedTensorMap, α::Number, β::Number)
    return add(ty, tx, α, β)
end

function VI.add!!(ty::ParametrisedTensorMap, tx::AbstractTensorMap, α::Number, β::Number)
    return add!(ty, tx, α, β)
end

function VI.add!!(ty::ParametrisedTensorMap, tx::ParametrisedTensorMap, α::Number, β::Number)
    return add!(ty, tx, α, β)
end

# inner
# -----
function VI.inner(ty::ParametrisedTensorMap, tx::AbstractTensorMap)
    throw(ErrorException("inner with ParametrisedTensorMap would retrun a function, thus this is not implemented"))
end

function VI.inner(ty::AbstractTensorMap, tx::ParametrisedTensorMap)
    throw(ErrorException("inner with ParametrisedTensorMap would retrun a function, thus this is not implemented"))
end

function VI.inner(ty::ParametrisedTensorMap, tx::ParametrisedTensorMap)
    throw(ErrorException("inner with ParametrisedTensorMap would retrun a function, thus this is not implemented"))
end
