import TensorOperations.tensorcontract!
import LinearAlgebra.mul!
import LinearAlgebra.lmul!

struct ParametrisedTensorMap{S,N1,N2,T<:AbstractTensorMap{S,N1,N2}} <: AbstractTensorMap{S,N1,N2}
    tensors::Vector{T}
    coeffs::Vector{Union{Number, Function}}
end

function ParametrisedTensorMap(tensor::T, coeff::C) where {S,N1,N2,T<:AbstractTensorMap{S,N1,N2},C<:Union{Number, Function}}
    return ParametrisedTensorMap{S,N1,N2,T}([tensor], [coeff])
end

function ParametrisedTensorMap(tensors::Vector{T}, coeffs::Vector{Union{<:Number, <:Function}}) where {S,N1,N2,T<:AbstractTensorMap{S,N1,N2}}
    return ParametrisedTensorMap{S,N1,N2,T}(tensors, coeffs)
end

function ParametrisedTensorMap(tensors::Vector{T}, coeffs::Vector{Any}) where {S,N1,N2,T<:AbstractTensorMap{S,N1,N2}}
    # check if the coeffs are only numbers and functions, then stuff them in a vector{number, function} if not, give error
    if all(x -> x isa Union{Number, Function}, coeffs)
        coeffVector = Vector{Union{Number, Function}}(coeffs)
        return ParametrisedTensorMap{S,N1,N2,T}(tensors, coeffVector)
    else
        throw(ArgumentError("coefficients must be a vector of numbers or functions (or a mix)"))
    end
end

function ParametrisedTensorMap(tensor::T) where {T<:AbstractTensorMap}
    return ParametrisedTensorMap([tensor], [1])
end

function ParametrisedTensorMap(tensors::Vector{T}) where {T<:AbstractTensorMap}
    return ParametrisedTensorMap(tensors, fill(1, length(tensors)))
end

function Base.show(io::IO, ptm::ParametrisedTensorMap)
    subscript(i) = join(Char(0x2080 + d) for d in reverse!(digits(i)))

    print(io, "ParametrisedTensorMap: ")
    for (i, tensor) in enumerate(ptm.tensors)
        if ptm.coeffs[i] isa Function
            print(io, "f", subscript(i))
        else
            print(io, "α", subscript(i))
        end
        print(io, "T", subscript(i))
        if i < length(ptm.tensors)
            print(io, " + ")
        end
    end
end

TensorKit.domain(t::ParametrisedTensorMap) = domain(t.tensors[1])
TensorKit.codomain(t::ParametrisedTensorMap) = codomain(t.tensors[1])

TensorKit.storagetype(::Type{<:ParametrisedTensorMap{S,N1,N2,T}}) where {S,N1,N2,T} = TensorKit.storagetype(T)

function (T::ParametrisedTensorMap)(t::Number)
    evaluated = Vector{typeof(T.tensors[1])}(undef, length(T.tensors))
    for i in eachindex(T.tensors)
        if typeof(T.coeffs[i]) <: Function
            evaluated[i] = T.coeffs[i](t) * T.tensors[i]
        else
            evaluated[i] = T.coeffs[i] * T.tensors[i]
        end
    end
    return sum(evaluated)
end