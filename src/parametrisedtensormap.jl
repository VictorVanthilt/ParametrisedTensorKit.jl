struct ParametrisedTensorMap{E,S,N1,N2,T<:AbstractTensorMap{E,S,N1,N2}} <: AbstractTensorMap{E,S,N1,N2}
    tensors::Vector{T}
    coeffs::Vector{Union{Number,Function}}
    function ParametrisedTensorMap{E,S,N1,N2,T}(tensors::Vector{T}, coeffs::Vector{Union{Number,Function}}) where {E,S,N1,N2,T<:AbstractTensorMap{E,S,N1,N2}}
        # check if tensors and coefficients are non zero
        newtensors = similar(tensors, 0)
        newcoeffs = similar(coeffs, 0)
        for i in eachindex(tensors)
            if coeffs[i] isa Number && !iszero(coeffs[i])
                if norm(tensors[i]) > eps(real(scalartype(tensors[i])))^(3 / 4)
                    push!(newtensors, tensors[i])
                    push!(newcoeffs, coeffs[i])
                end
            else # coeff is a function
                if norm(tensors[i]) > eps(real(scalartype(tensors[i])))^(3 / 4)
                    push!(newtensors, tensors[i])
                    push!(newcoeffs, coeffs[i])
                end
            end
        end
        if isempty(newtensors) # have at least one tensor stored, even if it and/or its coeff are zero
            push!(newtensors, zerovector(tensors[1]))
            push!(newcoeffs, 0)
        end
        return new{E,S,N1,N2,T}(newtensors, newcoeffs)
    end
end

# Constructors
# ------------

function ParametrisedTensorMap(tensor::T, coeff::C) where {E,S,N1,N2,T<:AbstractTensorMap{E,S,N1,N2},C<:Union{Number,Function}}
    tensorvec = Vector{T}(undef, 1)
    coeffvec = Vector{Union{Number, Function}}(undef, 1)
    tensorvec[1] = tensor
    coeffvec[1] = coeff
    return ParametrisedTensorMap{E,S,N1,N2,T}(tensorvec, coeffvec)
end

function ParametrisedTensorMap(tensors::Vector{T}, coeffs::Vector{Union{<:Number,<:Function}}) where {E,S,N1,N2,T<:AbstractTensorMap{E,S,N1,N2}}
    return ParametrisedTensorMap{E,S,N1,N2,T}(tensors, coeffs)
end

function ParametrisedTensorMap(tensors::Vector{T}, coeffs::Vector{<:Any}) where {E,S,N1,N2,T<:AbstractTensorMap{E,S,N1,N2}}
    # check if the coeffs are only numbers and functions, then stuff them in a vector{number, function} if not, give error
    if all(x -> x isa Union{Number,Function}, coeffs)
        coeffVector = Vector{Union{Number,Function}}(coeffs)
        return ParametrisedTensorMap{E,S,N1,N2,T}(tensors, coeffVector)
    else
        throw(ArgumentError("coefficients must be a vector of numbers or functions (or a mix)"))
    end
end

function ParametrisedTensorMap(tensor::T) where {T<:AbstractTensorMap}
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

    # if its empty, just print the type
    if isempty(ptm.tensors)
        print(io, "empty ParametrisedTensorMap{", eltype(ptm.tensors), "}")
        return
    end

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
    print(io, " | ")
    print(io, codomain(ptm.tensors[1]))
    print(io, " ← ")
    print(io, domain(ptm.tensors[1]))
end

# Parameter evaluation
# --------------------
function (ptm::ParametrisedTensorMap)(t::Number)
    evaluated = zerovector(ptm.tensors[1])
    for i in eachindex(ptm)
        if ptm.coeffs[i] isa Function
            axpby!(ptm.coeffs[i](t), ptm.tensors[i], 1, evaluated)
        else
            axpby!(ptm.coeffs[i], ptm.tensors[i], 1, evaluated)
        end
    end
    return evaluated
end

function eval_coeff(ptm::ParametrisedTensorMap, tval::Number)
    return ptm(tval)
end

# Coefficient combination
# -----------------------
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


# Addition methods
# ----------------
function Base.:+(t1::ParametrisedTensorMap, t2::ParametrisedTensorMap)
    newtensors = vcat(t1.tensors, t2.tensors)
    newcoeffs = vcat(t1.coeffs, t2.coeffs)
    return ParametrisedTensorMap(newtensors, newcoeffs)
end

function Base.:+(t1::ParametrisedTensorMap, t2::AbstractTensorMap)
    newtensors = vcat(t1.tensors, t2)
    newcoeffs = vcat(t1.coeffs, 1)
    return ParametrisedTensorMap(newtensors, newcoeffs)
end

# Code duplication is necessary here
# it's prettier if the order of α's and f's corresponds to the order of the tensors in a sum
function Base.:+(t1::AbstractTensorMap, t2::ParametrisedTensorMap)
    newtensors = vcat(t1, t2.tensors)
    newcoeffs = vcat(1, t2.coeffs)
    return ParametrisedTensorMap(newtensors, newcoeffs)
end

# Multiplication methods
# ----------------------
function Base.:*(α::Number, t::ParametrisedTensorMap)
    newcoeffs = Vector{Union{Number,Function}}(undef, length(t))
    for i in eachindex(t.coeffs)
        newcoeffs[i] = combinecoeff(α, t.coeffs[i])
    end
    return ParametrisedTensorMap(t.tensors, newcoeffs)
end

function Base.:*(t::ParametrisedTensorMap, α::Number)
    newcoeffs = map(t.coeffs) do x
        return combinecoeff(x, α)
    end
    return ParametrisedTensorMap(t.tensors, newcoeffs)
end

function Base.:*(f::Function, t::ParametrisedTensorMap)
    newcoeffs = map(t.coeffs) do x
        return combinecoeff(f, x)
    end
    typeof(newcoeffs) == Vector{Union{Number,Function}} || convert(Vector{Union{Number,Function}}, newcoeffs)
    return ParametrisedTensorMap(t.tensors, newcoeffs)
end

function Base.:*(t::ParametrisedTensorMap, f::Function)
    newcoeffs = map(t.coeffs) do x
        return combinecoeff(x, f)
    end
    return ParametrisedTensorMap(t.tensors, newcoeffs)
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
    ptms = Vector{ParametrisedTensorMap}(undef, length(t1))

    for i in 1:length(t1)
        tempTens = similar(t2.tensors, length(t2))
        tempCoeffs = similar(t2.coeffs, length(t2))

        for j in 1:length(t2)
            tempTens[j] = t1.tensors[i] * t2.tensors[j]
            tempCoeffs[j] = combinecoeff(t1.coeffs[i], t2.coeffs[j])
        end

        ptms[i] = ParametrisedTensorMap(tempTens, tempCoeffs)
    end
    return sum(ptms)
end

function adjoint(t::ParametrisedTensorMap)
    t.tensors = map(t.tensors) do x
        return adjoint(x)
    end
    t.coeffs = map(t.coeffs) do x
        return adjoint(x)
    end
    return t
end

function Base.convert(::Type{ParametrisedTensorMap}, t::AbstractTensorMap)
    return ParametrisedTensorMap(t)
end

function Base.convert(::Type{ParametrisedTensorMap{E,S,N1,N2,T}}, t::T) where {E,S,N1,N2,T<:AbstractTensorMap{E,S,N1,N2}}
    return ParametrisedTensorMap(t)
end

Base.eachindex(t::ParametrisedTensorMap) = eachindex(t.tensors)

# very poor definition, supposed to only give an indication!
function LinearAlgebra.norm(t::ParametrisedTensorMap)
    nm = 0
    for i in eachindex(t)
        if t.coeffs[i] isa Number
            nm += norm(t.coeffs[i]) * norm(t.tensors[i])
        else
            nm += norm(t.tensors[i])
        end
    end
    return nm
end

# copy!
function Base.copy(t::ParametrisedTensorMap)
    return ParametrisedTensorMap(copy(t.tensors), copy(t.coeffs))
end

function delay(t::ParametrisedTensorMap, dt::Number)
    return ParametrisedTensorMap(t.tensors, map(t.coeffs) do x
        if x isa Function
            return (t) -> x(t - dt)
        else
            return x
        end
    end)
end