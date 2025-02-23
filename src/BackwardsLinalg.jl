module BackwardsLinalg

using ChainRulesCore; import ChainRulesCore: rrule
using LinearAlgebra; import LinearAlgebra: ldiv!

struct ZeroAdder end
Base.:+(a, zero::ZeroAdder) = a
Base.:+(zero::ZeroAdder, a) = a
Base.:-(a, zero::ZeroAdder) = a
Base.:-(zero::ZeroAdder, a) = -a
Base.:-(zero::ZeroAdder) = zero


include("chainrules.jl")

include("qr.jl")
include("svd.jl")
include("lstsq.jl")
include("rsvd.jl")
include("symeigen.jl")
include("analy_func.jl")
include("cls.jl")
include("det.jl")
include("inv.jl")
include("lneq.jl")
include("lp.jl")
include("lp.jl")
include("sdp.jl")
include("lu.jl")
include("mxmul.jl")
include("scha_norm.jl")


end
