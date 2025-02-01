# Lazily multiply functions together
struct CoefficientFunction
    fs::Vector{Function}
end

const CF = CoefficientFunction

Base.:*(cf1::CF, cf2::CF) = CF(vcat(cf1.fs, cf2.fs))
Base.:*(cf::CF, f::Function) = CF(vcat(cf.fs, [f]))
Base.:*(f::Function, cf::CF) = CF(vcat([f], cf.fs))

Base.adjoint(cf::CF) = CF(map(cf.fs) do f
    return adjoint(f)
end)

function (cf::CF)(t::Number)
    p = 1
    for f in cf.fs
        p *= f(t)
    end
    return p
end