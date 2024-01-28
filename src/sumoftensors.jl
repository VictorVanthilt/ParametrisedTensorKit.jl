using MPSKit, TensorKit, TensorOperations
import LinearAlgebra.mul!

struct SumOfTensors{S,N1,N2,T<:AbstractTensorMap{S,N1,N2}} <: AbstractTensorMap{S,N1,N2}
    tensors::Vector{T}
end

function SumOfTensors(tens...)
    return SumOfTensors(collect(tens))
end

function Base.length(sot::SumOfTensors)
    return length(sot.tensors)
end

function Base.getindex(sot::SumOfTensors, i::Integer)
    return sot.tensors[i]
end

TensorKit.domain(t::SumOfTensors) = domain(t.tensors[1])
TensorKit.codomain(t::SumOfTensors) = codomain(t.tensors[1])

TensorKit.storagetype(::Type{<:SumOfTensors{S,N1,N2,T}}) where {S,N1,N2,T} = return TensorKit.storagetype(T)

function (sot::SumOfTensors)(t)
    evaluated = map(sot.tensors) do x
        if x isa ParametrisedTensorMap
            return x(t).coeff * x(t).tensor
        else
            return x
        end
    end
    return sum(evaluated)
end

# Addition methods
function Base.:+(sot1::SumOfTensors, sot2::SumOfTensors)
    return SumOfTensors(vcat(sot1.tensors, sot2.tensors))
end

Base.:+(t::ParametrisedTensorMap, sot::SumOfTensors) = SumOfTensors(vcat(t, sot.tensors))

Base.:+(sot::SumOfTensors, t::ParametrisedTensorMap) = SumOfTensors(vcat(t, sot.tensors))

Base.:+(t1::ParametrisedTensorMap, t2::ParametrisedTensorMap) = SumOfTensors(t1, t2)

Base.:+(t1::ParametrisedTensorMap, t2::AbstractTensorMap) = SumOfTensors(t1, t2)

Base.:+(t1::AbstractTensorMap, t2::ParametrisedTensorMap) = SumOfTensors(t1, t2)

# Multiplication methods
function mul!(C::AbstractTensorMap, A::AbstractTensorMap, B::SumOfTensors, α::Number, β::Number)
    println("here")
    mul!(C, A, B.tensors[1], α, β)
    println("passed")
    for i in 2:length(B)
        mul!(C, A, B.tensors[i], α, true)
    end
    return C
end

function mul!(C::AbstractTensorMap, A::SumOfTensors, B::AbstractTensorMap, α::Number, β::Number)
    println("here")
    mul!(C, A[1], B, α, β)
    println("passed")
    for i in 2:length(B)
        mul!(C, A, B.tensors[i], α, true)
    end
    return C
end