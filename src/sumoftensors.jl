using MPSKit, TensorKit, TensorOperations
import LinearAlgebra.mul!

#TODO: check compatibility of tensors at construction

struct SumOfTensors{S,N1,N2,T<:AbstractTensorMap{S,N1,N2}} <: AbstractTensorMap{S,N1,N2}
    tensors::Vector{T}
end

function SumOfTensors(tens...)
    return SumOfTensors(collect(tens))
end

function Base.show(io::IO, sot::SumOfTensors)
    subscript(i) = join(Char(0x2080 + d) for d in reverse!(digits(i)))

    print(io, "SumOfTensors: ")
    for (i, tensor) in enumerate(sot.tensors)
        if tensor isa ParametrisedTensorMap
            print(io, "α", subscript(i))
            print(io, "T", subscript(i))
        else
            print(io, "T", subscript(i))
        end
        if i < length(sot.tensors)
            print(io, " + ")
        end
    end
end

function Base.length(sot::SumOfTensors)
    return length(sot.tensors)
end

function Base.getindex(sot::SumOfTensors, i::Integer)
    return sot.tensors[i]
end

function eval_coeff(sot::SumOfTensors, tval)
    return sot(tval)
end

TensorKit.domain(t::SumOfTensors) = domain(t.tensors[1])
TensorKit.codomain(t::SumOfTensors) = codomain(t.tensors[1])

TensorKit.storagetype(::Type{<:SumOfTensors{S,N1,N2,T}}) where {S,N1,N2,T} = return TensorKit.storagetype(T)

function MPSKit.ismpoidentity(::SumOfTensors)
    return false
end

function (sot::SumOfTensors)(t)
    evaluated = map(sot.tensors) do x
        if x isa ParametrisedTensorMap
            return x(t)
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

Base.:+(t::ParametrisedTensorMap, sot::SumOfTensors) = SumOfTensors(t, sot.tensors...)

Base.:+(sot::SumOfTensors, t::ParametrisedTensorMap) = SumOfTensors(sot.tensors..., t)

Base.:+(t::AbstractTensorMap, sot::SumOfTensors) = SumOfTensors(t, sot.tensors...)

Base.:+(sot::SumOfTensors, t::AbstractTensorMap) = SumOfTensors(sot.tensors..., t)

Base.:+(t1::ParametrisedTensorMap, t2::ParametrisedTensorMap) = SumOfTensors(t1, t2)

Base.:+(t1::ParametrisedTensorMap, t2::AbstractTensorMap) = SumOfTensors(t1, t2)

Base.:+(t1::AbstractTensorMap, t2::ParametrisedTensorMap) = SumOfTensors(t1, t2)



# multiplication methods
function Base.:*(t1::AbstractTensorMap, t2::SumOfTensors)
    return SumOfTensors(map(t2.tensors) do x
        return t1 * x
    end...)
end

function Base.:*(t1::SumOfTensors, t2::AbstractTensorMap)
    return SumOfTensors(map(t1.tensors) do x
        return x * t2
    end...)
end

# ======================
# mul! methods
# ======================
function mul!(C::AbstractTensorMap, A::AbstractTensorMap, B::SumOfTensors, α::Number, β::Number)
    C = ParametrisedTensorMap(C, β)
    tensors = Array{Any}(missing, length(B))
    for i in 1:length(B)
        tensors[i] = α * A * B[i]
    end
    return SumOfTensors(tensors...)
end

function mul!(C::AbstractTensorMap, A::SumOfTensors, B::AbstractTensorMap, α::Number, β::Number)
    C = ParametrisedTensorMap(C, β)
    tensors = Array{Any}(missing, length(A))
    for i in 1:length(A)
        tensors[i] = α * A[i] * B
    end
    return SumOfTensors(tensors...)
end