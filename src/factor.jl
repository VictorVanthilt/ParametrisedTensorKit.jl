struct factor{T<:Union{Number, Function}} <: Number 
    nargs::Int
    f::T

    function factor(nargs::Int, f::T) where T<:Union{Number, Function}
        # nargs must be a positive integer
        nargs >= 0 || throw(DomainError("The number of arguments must be a positive integer (>= 0)"))
        new{T}(nargs, f)
    end
end

function factor(a::Int)
    return factor(0, a)
end

function factor(a::Function)
    if length(methods(a)) > 1
        throw(DomainError("Cannot create a factor from a function with multiple methods, explicitly include number of arguments in the factor constructor"))
    end

    return factor(methods(a)[1].nargs - 1, a)
end

function Base.show(io::IO, f::factor)
    print(io, "Factor with $(f.nargs) arguments")
end

# Combining two factors
# If one function has less arguments than the other, the function with less arguments only takes the first n arguments of the function with more arguments
function Base.:*(a::factor, b::factor)

    # Check if they are both functions
    if (isa(a.f, Function) && isa(b.f, Function))
        (a.nargs != b.nargs) && @warn "Combining factors with different number of arguments ($(a.nargs) != $(b.nargs))\n Taking the first $(min(a.nargs, b.nargs)) arguments of the function with more arguments"
        nargs = max(a.nargs, b.nargs)
        f = (x...) -> a.f(x[1:a.nargs]...) * b.f(x[1:b.nargs]...)
        return factor(nargs, f)
    end

    # Check if they are both numbers
    if (isa(a.f, Number) && isa(b.f, Number))
        return factor(a.nargs, a.f * b.f)
    end

    # Check if one of them is a number
    if (isa(a.f, Number))
        f = (x...) -> a.f * b.f(x...)
        return factor(a.nargs, f)
    end

    if (isa(b.f, Number))
        f = (x...) -> a.f(x...) * b.f
        return factor(a.nargs, f)
    end

    # Throw an error that this case is not supported
    throw(DomainError("Multiplication of the two factors is not supported"))
end

function Base.:*(a::factor, b::Number)
    return factor(a.nargs, a.f * b)
end

function Base.:*(a::Number, b::factor)
    return factor(b.nargs, a * b.f)
end

function Base.:*(a::factor, b::Function)
    throw(DomainError("Multiplication of a factor and a function is not supported, wrap your function in a factor"))
end

function Base.:*(a::Function, b::factor)
    throw(DomainError("Multiplication of a factor and a function is not supported, wrap your function in a factor"))
end

function (f::factor)(x...)
    # Consider if adding the check below hampers the performance
    # Check if the function has the correct number of arguments
    # Actually it adds not a lot of overhead with respect to the at times ~800 times slowdown of the function call. Why is it THAT slow?
    # length(x) == f.nargs || throw(DomainError(x, "The factor takes $(f.nargs) arguments, but $(length(x)) have been given."))
    return f.f(x...)
end

# Tests
f1 = factor(1, (x) -> sin(x))
f2 = factor(2, (x, y) -> x * y)

f1 * f1
f2 * f2
f1f2 = f1 * f2

f5 = factor(2)

f1 * f5
f5 * f1


factor((t) -> sin(t))
func = (t) -> sin(t)
s_t(t) = sin(t)
c_t(t) = cos(t)
cs = factor(s_t) * factor(c_t)
sinfactor = factor(func)


using BenchmarkTools
@btime f1f2(1.0, 2.0)
@btime sin(1)

