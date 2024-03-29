import TensorOperations.tensorcontract!
import LinearAlgebra.mul!
import LinearAlgebra.lmul!

struct ParametrisedTensorMap{S,N1,N2,T<:AbstractTensorMap{S,N1,N2}} <: AbstractTensorMap{S,N1,N2}
    tensor::T
    coeff
end

function ParametrisedTensorMap(tensor::T, coeff) where {S,N1,N2,T<:AbstractTensorMap{S,N1,N2}}
    return ParametrisedTensorMap{S,N1,N2,T}(tensor, coeff)
end

function ParametrisedTensorMap(tensor::T) where {T<:AbstractTensorMap}
    return ParametrisedTensorMap(tensor, 1)
end

TensorKit.domain(t::ParametrisedTensorMap) = domain(t.tensor)
TensorKit.codomain(t::ParametrisedTensorMap) = codomain(t.tensor)

function (T::ParametrisedTensorMap)(t::Number)
    return T.coeff(t) * T.tensor
end

TensorKit.storagetype(::Type{<:ParametrisedTensorMap{S,N1,N2,T}}) where {S,N1,N2,T} = TensorKit.storagetype(T)

function Base.show(io::IO, t::ParametrisedTensorMap)
    print(io, "ParametrisedTensorMap: ")
    print(io, "αT")
    print(io, " where α = ", t.coeff)
end

# Multiplication methods
function Base.:*(t1::ParametrisedTensorMap, t2::ParametrisedTensorMap)
    newtens = t1.tensor * t2.tensor
    return ParametrisedTensorMap(newtens, combinecoeff(t1.coeff, t2.coeff))
end

function Base.:*(t1::ParametrisedTensorMap, t2::AbstractTensorMap)
    newtens = t1.tensor * t2
    return ParametrisedTensorMap(newtens, combinecoeff(t1.coeff, one(promote_type(scalartype(t1), scalartype(t2)))))
end

function Base.:*(t1::AbstractTensorMap, t2::ParametrisedTensorMap)
    newtens = t1 * t2.tensor
    return ParametrisedTensorMap(newtens, combinecoeff(one(promote_type(scalartype(t1), scalartype(t2))), t2.coeff))
end

function Base.:*(N::Number, t::ParametrisedTensorMap)
    return ParametrisedTensorMap(N * t.tensor, t.coeff)
end

function Base.:*(t::ParametrisedTensorMap, N::Number)
    return ParametrisedTensorMap(t.tensor, combinecoeff(t.coeff, N))
end

function Base.:*(f::Function, t::ParametrisedTensorMap)
    return ParametrisedTensorMap(t.tensor, combinecoeff(f, t.coeff))
end

function Base.:*(t::ParametrisedTensorMap, f::Function)
    return ParametrisedTensorMap(t.tensor, combinecoeff(t.coeff, f))
end

function Base.:*(f::Function, t::AbstractTensorMap)
    return ParametrisedTensorMap(t, f)
end

function Base.:*(t::AbstractTensorMap, f::Function)
    return ParametrisedTensorMap(t, f)
end

function Base.:^(t::ParametrisedTensorMap, n::Integer)
    return ParametrisedTensorMap(t.tensor^n, t -> t.coeff(t)^n)
end

# Addition methods
function Base.:+(t1::ParametrisedTensorMap, t2::ParametrisedTensorMap)
    return SumOfTensors(t1, t2)
end

# Combining coefficients with eachother
function combinecoeff(f1::Function, f2::Number)
    return (t) -> f1(t) * f2
end

function combinecoeff(f1::Number, f2::Function)
    return (t) -> f1 * f2(t)
end

function combinecoeff(f1::Function, f2::Function)
    return (t) -> f1(t) * f2(t)
end

function combinecoeff(f1::Number, f2::Number)
    return f1 * f2
end

function MPSKit.ismpoidentity(::ParametrisedTensorMap)
    return false
end

function eval_coeff(t::ParametrisedTensorMap, tval)
    return t(tval)
end

TensorKit.has_shared_permute(t::ParametrisedTensorMap, args...) = false

function TensorKit.similar(t::ParametrisedTensorMap, T::Type, P::TensorMapSpace)
    tens = similar(t.tensor, T, P)
    return ParametrisedTensorMap(tens)
end

function adjoint(t::ParametrisedTensorMap)
    return ParametrisedTensorMap(TensorKit.adjoint(t.tensor), adjoint(t.coeff))
end

function convert(::Type{ParametrisedTensorMap}, t::AbstractTensorMap)
    return ParametrisedTensorMap(t)
end

# ======================
# tensorcontract methods
# ======================
function tensorcontract!(C::AbstractTensorMap{S,N₁,N₂}, pAB::Index2Tuple,
                                          A::ParametrisedTensorMap{S,N₁,N₂,T}, pA::Index2Tuple, conjA::Symbol,
                                          B::AbstractTensorMap{S}, pB::Index2Tuple, conjB::Symbol,
                                          α::Number, β::Number) where {S,N₁,N₂,T}
    newalpha = combinecoeff(α, A.coeff)
    return tensorcontract!(C, pAB, A.tensor, pA, conjA, B, pB, conjB, newalpha, β)
end

function tensorcontract!(C::AbstractTensorMap{S,N₁,N₂}, pAB::Index2Tuple,
                                          A::AbstractTensorMap{S}, pA::Index2Tuple, conjA::Symbol,
                                          B::ParametrisedTensorMap, pB::Index2Tuple, conjB::Symbol,
                                          α::Number, β::Number) where {S,N₁,N₂}
    newalpha = combinecoeff(α, B.coeff)
    return tensorcontract!(C, pAB, A, pA, conjA, B.tensor, pB, conjB, newalpha, β)
end

function tensorcontract!(C::AbstractTensorMap{S,N₁,N₂}, pAB::Index2Tuple,
                                          A::ParametrisedTensorMap, pA::Index2Tuple, conjA::Symbol,
                                          B::ParametrisedTensorMap, pB::Index2Tuple, conjB::Symbol,
                                          α::Number, β::Number) where {S,N₁,N₂}
    newalpha = combinecoeff(α, combinecoeff(A.coeff, B.coeff))
    return tensorcontract!(C, pAB, A.tensor, pA, conjA, B.tensor, pB, conjB, newalpha, β)
end

# promote one of the AbstractTensorMaps to a ParametrisedTensorMap and add the coeff
function tensorcontract!(C::AbstractTensorMap{S,N₁,N₂}, pAB::Index2Tuple,
                                          A::AbstractTensorMap{S}, pA::Index2Tuple, conjA::Symbol,
                                          B::AbstractTensorMap{S}, pB::Index2Tuple, conjB::Symbol,
                                          α::Function, β::Number) where {S,N₁,N₂}
    newC = deepcopy(C)
    tensorcontract!(C, pAB, A, pA, conjA, B, pB, conjB, 1, 0)
    return SumOfTensors(β*newC, ParametrisedTensorMap(C, α))
end

# ======================
# mul! methods
# ======================
function mul!(C::AbstractTensorMap, A::ParametrisedTensorMap, B::AbstractTensorMap, α::Number, β::Number)
    newalpha = combinecoeff(α, A.coeff)
    return mul!(C, A.tensor, B, newalpha, β)
end

function mul!(C::AbstractTensorMap, A::AbstractTensorMap, B::ParametrisedTensorMap, α::Number, β::Number)
    newalpha = combinecoeff(α, B.coeff)
    return mul!(C, A, B.tensor, newalpha, β)
end

function mul!(C::AbstractTensorMap, A::ParametrisedTensorMap, B::ParametrisedTensorMap, α::Number, β::Number)
    newalpha = combinecoeff(α, combinecoeff(A.coeff, B.coeff))
    return mul!(C, A.tensor, B.tensor, newalpha, β)
end

function lmul!(α::Number, t::ParametrisedTensorMap)
    return ParametrisedTensorMap(t.tensor, combinecoeff(α, t.coeff))
end