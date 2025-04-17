# Lazily multiply functions together
mutable struct CoefficientFunction
    fs::Vector{Function}
end

const CF = CoefficientFunction

Base.length(cf::CF) = length(cf.fs)

Base.:*(cf1::CF, cf2::CF) = CF(vcat(cf1.fs, cf2.fs))
Base.:*(cf::CF, f::Function) = CF(vcat(cf.fs, [f]))
Base.:*(f::Function, cf::CF) = CF(vcat([f], cf.fs))
Base.:*(cf::CF, α::Number) = α*cf

function Base.:*(α::Number, cf::CF) # this modifies the original object and it shouldn't
    f = cf.fs[1]
    cf.fs[1] = (t) -> α*f(t)
    return cf
end

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

# TODO: move this to MPSKit
# function integrate(cf::CF, t₀::Number, t₁::Number)
#     return prod(map(cf.fs) do f
#                     return quadgk(f, t₀, t₁)[1]
#                 end)
# end