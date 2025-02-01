module ParametrisedTensorKit

# TODO: don't import MPSKit as a whole, just get the types needed for time-evaluation
using MPSKit, TensorOperations, BlockTensorKit, TensorKit, LinearAlgebra, VectorInterface
using QuadGK

export ParametrisedTensorMap, delay, integrate

using TensorOperations: AbstractBackend

import VectorInterface as VI
import TensorOperations as TO
import TensorKit as TK

include("coefficients.jl")
include("parametrisedtensormap.jl")
include("MPSKit.jl")
include("TensorOperations.jl")
include("VectorInterface.jl")
include("TensorKit.jl")
include("linalg.jl")

end