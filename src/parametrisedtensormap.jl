struct ParametrisedTensorMap{E,S,N1,N2,T<:AbstractTensorMap{E,S,N1,N2}} <: AbstractTensorMap{E,S,N1,N2}
    tensors::Vector{T}
    coeffs::Vector{Union{Number,Function}}
    function ParametrisedTensorMap{E,S,N1,N2,T}(tensors::Vector{T}, coeffs::Vector{Union{Number,Function}}) where {E,S,N1,N2,T<:AbstractTensorMap{E,S,N1,N2}}
        newtensors = similar(tensors, 0)
        newcoeffs = similar(coeffs, 0)
        has_constant = false
        for i in eachindex(tensors)
            if norm(tensors[i]) > eps(real(scalartype(tensors[i])))^(3 / 4) # Check if it is worth to store the tensor
                if coeffs[i] isa Number && !iszero(coeffs[i]) # Insert constant multplied tensors at the front or add them to existing constant tensor
                    if has_constant
                        newtensors[1] += coeffs[i] * tensors[i]
                    else
                        insert!(newtensors, 1, coeffs[i] * tensors[i])
                        insert!(newcoeffs, 1, 1)
                        has_constant = true
                    end
                else # Coeff is a function and thus needs to be stored independently
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

# use getproperty to check if the PTM has a constant tensor instead of storing a bool as a field
function Base.getproperty(t::ParametrisedTensorMap, prop::Symbol)
    if prop == :has_constant
        return t.coeffs[1] isa Number
    end
    return getfield(t, prop)
end

# Constructors
# ------------

function ParametrisedTensorMap(tensor::T, coeff::C) where {E,S,N1,N2,T<:AbstractTensorMap{E,S,N1,N2},C<:Union{Number,Function}}
    tensorvec = Vector{T}(undef, 1)
    coeffvec = Vector{Union{Number,Function}}(undef, 1)
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
    if tensor isa ParametrisedTensorMap
        return tensor
    end
    return ParametrisedTensorMap(tensor, 1)
end

function ParametrisedTensorMap(tensors::Vector{T}) where {T<:AbstractTensorMap}
    return ParametrisedTensorMap(tensors, fill(1, length(tensors)))
end

function ParametrisedTensorMap{E}(::UndefInitializer, TMS::TensorMapSpace) where E
    N2 = numin(TMS)
    N1 = numout(TMS)
    S = spacetype(TMS)
    T = TensorMap{E,S,N1,N2,Vector{E}}
    
    tensors = Vector{T}(undef, 1)
    tensors[1] = zeros(E, TMS)

    coeffs = Vector{Union{Number,Function}}(undef, 1)
    coeffs[1] = 0

    return ParametrisedTensorMap{E,S,N1,N2,T}(tensors, coeffs)
end

function ParametrisedTensorMap{E}(tensors::Vector{<:AbstractTensorMap{E}}, coeffs::Vector{Union{Number, Function}}) where E
    return ParametrisedTensorMap(tensors, coeffs)
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

function eval_coeff(F::Union{Number,Function}, t::Number)
    return F isa Number ? F : F(t)
end

function eval_coeffs(ptm::ParametrisedTensorMap, t::Number)
    return ptm(t)
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

Base.:+(t1::AbstractTensorMap, t2::ParametrisedTensorMap) = t2 + t1

# Multiplication methods
# ----------------------
function Base.:*(α::Number, t::ParametrisedTensorMap)
    newcoeffs = map(t.coeffs) do x
        return combinecoeff(x, α)
    end
    return ParametrisedTensorMap(t.tensors, newcoeffs)
end

Base.:*(t::ParametrisedTensorMap, α::Number) = α * t

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
    newtensors = similar(t1.tensors, length(t1) * length(t2))
    newcoeffs = Vector{Union{Number,Function}}(undef, length(t1) * length(t2))
    for i in eachindex(t1)
        for j in eachindex(t2)
            index = (i - 1) * length(t2) + j
            newtensors[index] = t1.tensors[i] * t2.tensors[j]
            newcoeffs[index] = combinecoeff(t1.coeffs[i], t2.coeffs[j])
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

function Base.similar(t::ParametrisedTensorMap)
    return ParametrisedTensorMap{scalartype(t)}(undef, space(t))
end
function Base.similar(t::ParametrisedTensorMap, TMS::TensorMapSpace)
    return ParametrisedTensorMap{scalartype(t)}(undef, TMS)
end

# copy!
function Base.copy(t::ParametrisedTensorMap)
    return ParametrisedTensorMap(copy(t.tensors), copy(t.coeffs))
end

# Delay the coefficients of a ParametrisedTensorMap, going back in time by dt
function delay(t::ParametrisedTensorMap, dt::Number)
    return ParametrisedTensorMap(t.tensors, map(t.coeffs) do x
        if x isa Function
            return (t) -> x(t - dt)
        else
            return x
        end
    end)
end