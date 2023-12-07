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
    # return PTM.coeff(t) * PTM.tensor
    # alternative:
    return ParametrisedTensorMap(PTM.tensor, PTM.coeff(t))
end

TensorKit.storagetype(::Type{<:ParametrisedTensorMap{S,N1,N2,T}}) where {S,N1,N2,T} = TensorKit.storagetype(T)

TensorKit.block(t::ParametrisedTensorMap{S,N1,N2,T,E}, ::Trivial) where {S,N1,N2,T, E<:Number} = TensorKit.block(t.tensor, Trivial()) * t.coeff

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

function Base.:^(t::ParametrisedTensorMap, n::Integer)
    newtens = t.tensor^n
    return ParametrisedTensorMap(newtens, exponentiatecoeff(t.coeff, n))
end

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

function exponentiatecoeff(f::Function, n::Integer)
    return (t) -> f(t)^n
end

function MPSKit.ismpoidentity(::ParametrisedTensorMap)
    return false
end

function eval_coeff(t::ParametrisedTensorMap, tval)
    return t(tval)
end
eval_coeff(t::Any, tval) = t

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