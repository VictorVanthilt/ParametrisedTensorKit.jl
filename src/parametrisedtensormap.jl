struct ParametrisedTensorMap{E,S,N1,N2,T<:AbstractTensorMap{E,S,N1,N2}} <: AbstractTensorMap{E,S,N1,N2}
    tensors::Vector{T}
    coeffs::Vector{Prefactor}
    function ParametrisedTensorMap{E,S,N1,N2,T}(tensors::Vector{T}, coeffs::Vector{Prefactor}) where {E,S,N1,N2,T}
        @assert length(tensors) == length(coeffs) "The amount of tensors and coefficients must be the same"
        return new{E,S,N1,N2,T}(tensors, coeffs)
    end
end

# Constructors
# ------------
function ParametrisedTensorMap(tensor::T, coeff::Prefactor) where {E,S,N1,N2,T<:AbstractTensorMap{E,S,N1,N2}}
    return ParametrisedTensorMap{E,S,N1,N2,T}(Vector{T}[tensor], Vector{Prefactor}[coeff])
end
function ParametrisedTensorMap(tensor::T, coeff::Nunction) where {E,S,N1,N2,T<:AbstractTensorMap{E,S,N1,N2}}
    return ParametrisedTensorMap{E,S,N1,N2,T}([tensor], [Prefactor(coeff)])
end
function ParametrisedTensorMap(tensors::Vector{T}, coeffs::Vector{Prefactor}) where {E,S,N1,N2,T<:AbstractTensorMap{E,S,N1,N2}}
    return ParametrisedTensorMap{E,S,N1,N2,T}(tensors, coeffs)
end
function ParametrisedTensorMap(tensors::Vector{T}, coeffs::Vector{Nunction}) where {E,S,N1,N2,T<:AbstractTensorMap{E,S,N1,N2}}
    return ParametrisedTensorMap{E,S,N1,N2,T}(tensors, Vector{Prefactor}(Prefactor.(coeffs)))
end
function ParametrisedTensorMap(tensor::T) where {T<:AbstractTensorMap}
    if tensor isa ParametrisedTensorMap
        return tensor
    end
    return ParametrisedTensorMap(tensor, 1)
end
function ParametrisedTensorMap(tensors::Vector{T}) where {T<:AbstractTensorMap}
    return ParametrisedTensorMap(tensors, fill(1, length(tensors)))
end

Base.length(t::ParametrisedTensorMap) = length(t.tensors)

# Construct by multiplying coefficient function
function Base.:*(f::Function, t::AbstractTensorMap)
    return ParametrisedTensorMap(t, f)
end

function Base.:*(t::AbstractTensorMap, f::Function)
    return ParametrisedTensorMap(t, f)
end

# Show
# ----
function Base.show(io::IO, ptm::ParametrisedTensorMap)
    subscript(i) = join(Char(0x2080 + d) for d in reverse!(digits(i)))

    print(io, "ParametrisedTensorMap: ")
    for i in eachindex(ptm)
        print(io, "f", subscript(i))
        print(io, "T", subscript(i))
        if i < length(ptm.tensors)
            print(io, " + ")
        end
    end
    print(io, " | ")
    print(io, space(ptm))
end

# Parameter evaluation
# --------------------
function (ptm::ParametrisedTensorMap)(t::Number)
    evaluated = zerovector(ptm.tensors[1])
    for i in eachindex(ptm)
        axpby!(eval_coeff(ptm.coeffs[i], t), ptm.tensors[i], 1, evaluated)
    end

    return evaluated
end

eval_coeff(F::Prefactor, t::Number) = F(t)
eval_coeffs(ptm::ParametrisedTensorMap, t::Number) = ptm(t)

# Addition methods
# ----------------
function Base.:+(t1::ParametrisedTensorMap, t2::ParametrisedTensorMap)
    newtensors = vcat(deepcopy(t1.tensors), deepcopy(t2.tensors))
    newcoeffs = vcat(deepcopy(t1.coeffs), deepcopy(t2.coeffs))
    return ParametrisedTensorMap(newtensors, newcoeffs)
end

function Base.:+(t1::ParametrisedTensorMap, t2::AbstractTensorMap)
    if iszero(t2) # never add exact zeros!
        return deepcopy(t1)
    end
    newtensors = vcat(t1.tensors, t2)
    newcoeffs = vcat(t1.coeffs, Prefactor(1))
    return ParametrisedTensorMap(newtensors, newcoeffs)
end

Base.:+(t1::AbstractTensorMap, t2::ParametrisedTensorMap) = t2 + t1

# Multiplication methods
# ----------------------
# Massive code duplication for disambiguation, absorb numbers in the tensors
function Base.:*(α::Number, t::ParametrisedTensorMap)
    newtensors = map(t.tensors) do x
        return α*x
    end
    return ParametrisedTensorMap(newtensors, deepcopy(t.coeffs))
end
function Base.:*(α::Function, t::ParametrisedTensorMap)
    newcoeffs = map(t.coeffs) do x
        return α*x
    end
    return ParametrisedTensorMap(deepcopy(t.tensors), newcoeffs)
end

function Base.:*(t::ParametrisedTensorMap, α::Number)
    newtensors = map(t.tensors) do x
        return x*α
    end
    return ParametrisedTensorMap(newtensors, deepcopy(newcoeffs))
end
function Base.:*(t::ParametrisedTensorMap, α::Function)
    newcoeffs = map(t.coeffs) do x
        return x * α
    end
    return ParametrisedTensorMap(deepcopy(t.tensors), newcoeffs)
end

function Base.:*(t1::AbstractTensorMap, t2::ParametrisedTensorMap)
    newtensors = map(t2.tensors) do x
        return t1 * x
    end
    return ParametrisedTensorMap(newtensors, t2.coeffs)
end

function Base.:*(t1::ParametrisedTensorMap, t2::AbstractTensorMap)
    newtensors = map(t1.tensors) do x
        return x * t2
    end
    return ParametrisedTensorMap(newtensors, t1.coeffs)
end

function Base.:*(t1::ParametrisedTensorMap, t2::ParametrisedTensorMap)
    newtensors = similar(t1.tensors, length(t1) * length(t2))
    newcoeffs = Vector{Prefactor}(undef, length(t1) * length(t2))
    for i in eachindex(t1)
        for j in eachindex(t2)
            index = (i - 1) * length(t2) + j
            newtensors[index] = t1.tensors[i] * t2.tensors[j]
            newcoeffs[index] = t1.coeffs[i] * t2.coeffs[j]
        end
    end
    return ParametrisedTensorMap(newtensors, newcoeffs)
end

function Base.adjoint(t::ParametrisedTensorMap)
    newtensors = map(t.tensors) do x
        return adjoint(x)
    end
    newcoeffs = map(t.coeffs) do x
        if x isa Function
            return (t) -> adjoint(x(t))
        else
            return adjoint(x)
        end
    end
    return ParametrisedTensorMap(newtensors, newcoeffs)
end

function Base.convert(::Type{ParametrisedTensorMap}, t::AbstractTensorMap)
    return ParametrisedTensorMap(t)
end

function Base.convert(::Type{ParametrisedTensorMap{E,S,N1,N2,T}}, t::T) where {E,S,N1,N2,T<:AbstractTensorMap{E,S,N1,N2}}
    return ParametrisedTensorMap(t)
end

Base.eachindex(t::ParametrisedTensorMap) = eachindex(t.tensors)

# Make sure that similar returns a PTM with a similar amount of stored tensors
function Base.similar(t::ParametrisedTensorMap)
    return ParametrisedTensorMap(similar.(t.tensors), Prefactor.(zeros(length(t))))
end
function Base.similar(t::ParametrisedTensorMap, TMS::TensorMapSpace)
    return ParametrisedTensorMap(similar.(t.tensors, Ref(TMS)), Prefactor.(zeros(length(t))))
end
function Base.similar(t::ParametrisedTensorMap, E::Type{<:Number})
    return ParametrisedTensorMap(similar.(t.tensors, E), Prefactor.(zeros(E, length(t))))
end

# copy!
function Base.copy(t::ParametrisedTensorMap)
    return ParametrisedTensorMap(copy(t.tensors), copy(t.coeffs))
end

# Delay the coefficients of a ParametrisedTensorMap, going back in time by dt
# function delay(t::ParametrisedTensorMap, dt::Number)
#     return ParametrisedTensorMap(t.tensors, map(t.coeffs) do x
#         if x isa Function
#             return (t) -> x(t - dt)
#         else
#             return x
#         end
#     end)
# end

function purge!(t::ParametrisedTensorMap)
    to_keep = [!iszero(t.tensors[i]) for i in eachindex(t.tensors)]
    for (i, keep) in enumerate(to_keep)
        if !keep
            deleteat!(t.tensors, i)
            deleteat!(t.coeffs, i)
        end
    end
    return t 
end

purge!(t::AbstractTensorMap) = t
