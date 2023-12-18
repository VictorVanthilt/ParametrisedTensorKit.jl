using MPSKit, TensorKit, TensorOperations
const TO = TensorOperations
import TensorOperations.tensorcontract!

struct ParametrisedTensorMap{S,N1,N2,T<:AbstractTensorMap{S,N1,N2},E} <: AbstractTensorMap{S,N1,N2}
    tensor::T
    coeff::E
end

function ParametrisedTensorMap(tensor::T, coeff::E) where {S,N1,N2,T<:AbstractTensorMap{S,N1,N2},E}
    return ParametrisedTensorMap{S,N1,N2,T,E}(tensor, coeff)
end

function ParametrisedTensorMap(tensor::T) where {T<:AbstractTensorMap}
    return ParametrisedTensorMap(tensor, 1)
end

TensorKit.domain(t::ParametrisedTensorMap) = domain(t.tensor)
TensorKit.codomain(t::ParametrisedTensorMap) = codomain(t.tensor)

function (PTM::ParametrisedTensorMap)(t::Number)
    return ParametrisedTensorMap(PTM.tensor, PTM.coeff(t))
end

TensorKit.storagetype(::Type{<:ParametrisedTensorMap{S,N1,N2,T}}) where {S,N1,N2,T} = TensorKit.storagetype(T)

TensorKit.block(t::ParametrisedTensorMap{S,N1,N2,T,E}, ::Trivial) where {S,N1,N2,T, E<:Number} = TensorKit.block(t.tensor, Trivial()) * t.coeff

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
    return ParametrisedTensorMap(t.tensor, combinecoeff(N, t.coeff))
end

function Base.:*(t::ParametrisedTensorMap, N::Number)
    return ParametrisedTensorMap(t.tensor, combinecoeff(t.coeff, N))
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

function (H::MPOHamiltonian{T})(t) where {S,T<:BlockTensorMap{S,2,2,<:Union{MPSKit.MPOTensor, ParametrisedTensorMap, TensorKit.BraidingTensor}}}
    return MPOHamiltonian(map(H.data) do x
        new_subtensors = Dict(I => eval_coeff(old_subtensor, t) for (I, old_subtensor) in nonzero_pairs(x))
        new_tensortype = Union{(typeof.(values(new_subtensors)))...}

        newx = BlockTensorMap{S,2,2,new_tensortype}(undef, x.codom, x.dom)
        for (key, value) in new_subtensors
            newx[key] = value
        end
        return newx
    end)
end

TensorKit.has_shared_permute(t::ParametrisedTensorMap, args...) = false

function TensorKit.similar(t::ParametrisedTensorMap, T::Type, P::TensorMapSpace)
    tens = similar(t.tensor, T, P)
    return ParametrisedTensorMap(tens)
end

function adjoint(PTM::ParametrisedTensorMap)
    return ParametrisedTensorMap(TensorKit.adjoint(PTM.tensor), adjoint(PTM.coeff))
end

function convert(::Type{ParametrisedTensorMap}, t::AbstractTensorMap)
    return ParametrisedTensorMap(t)
end

# tensorcontract
function tensorcontract!(C::AbstractTensorMap{S,N₁,N₂}, pAB,
                                          A::ParametrisedTensorMap, pA, conjA::Symbol,
                                          B::AbstractTensorMap{S}, pB, conjB::Symbol,
                                          α::Number, β::Number) where {S,N₁,N₂}
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

function tensorcontract!(C::AbstractTensorMap{S,N₁,N₂}, pAB::Index2Tuple,
                         A::AbstractTensorMap{S}, pA::Index2Tuple, conjA::Symbol,
                         B::AbstractTensorMap{S}, pB::Index2Tuple, conjB::Symbol,
                         α::Function, β::Number) where {S,N₁,N₂}
    A = ParametrisedTensorMap(A, α)
    return tensorcontract!(C, pAB, A.tensor, pA, conjA, B, pB, conjB, 1, β)
end