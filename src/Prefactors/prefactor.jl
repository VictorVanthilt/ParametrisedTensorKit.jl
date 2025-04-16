const Nunction = Union{Number, Function}

struct Prefactor
    data::Vector{Nunction}
end

function Base.show(io::IO, pf::Prefactor)
    print(io, "Prefactor(")
    for i in 1:length(pf.data)
        if i > 1
            print(io, ", ")
        end
        print(io, pf.data[i])
    end
    print(io, ")")
end

Base.parent(pf::Prefactor) = pf.data
Base.eachindex(pf::Prefactor) = eachindex(parent(pf))
Base.iterate(pf::Prefactor) = iterate(parent(pf))
Base.iterate(pf::Prefactor, state) = iterate(parent(pf), state)
Base.length(pf::Prefactor) = length(parent(pf))

# Multiplication logic
Base.:*(pf::Prefactor, x::Nunction) = Prefactor(vcat(pf.data, x))
Base.:*(x::Nunction, pf::Prefactor) = Prefactor(vcat(x, pf.data))
Base.:*(pf1::Prefactor, pf2::Prefactor) = Prefactor(vcat(pf1.data, pf2.data))

# Function evaluation logic
function (pf::Prefactor)(t::Number)
    result = 1.0 + 0.0im
    for x in pf
        if x isa Function
            result *= x(t)
        elseif x isa Number
            result *= x
        else
            error("Unsupported type in Prefactor: ", typeof(x))
        end
    end
    return result
end