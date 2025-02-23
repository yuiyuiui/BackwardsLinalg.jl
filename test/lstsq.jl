using BackwardsLinalg
using Test, Random
using Zygote
import Mooncake, DifferentiationInterface


function gradient_check(f, args...; η = 1e-5)
    println(1)
    g = gradient(f, args...)
    println(2)
    dy_expect = η*sum(abs2.(g[1]))
    println(3)
    dy = f(args...)-f([gi === nothing ? arg : arg.-η.*gi for (arg, gi) in zip(args, g)]...)
    println(4)
    isapprox(dy, dy_expect, rtol=1e-2, atol=1e-8)
end

# test for real and complex
@testset "lstsq" begin
	TYPE = [Float64, ComplexF64]
	for T in TYPE
		Random.seed!(3)
		M, N = 10, 5
		A = randn(T, M, N)
		b = randn(T, M)
		op = randn(T, N, N)
		op += op'

		function tfunc(A, b)
			x = BackwardsLinalg.lstsq(A, b)
			return real(x' * op * x)
		end
		tfuncA(A) = tfunc(A, b)
		tfuncb(b) = tfunc(A, b)
        @show T
		@test gradient_check(tfuncA, A)
		@test gradient_check(tfuncb, b)
	end
end



@testset "mooncake" begin
    T = Float64
    Random.seed!(3)
    M, N = 10, 5
    A = randn(T, M, N)
    b = randn(T, M)
    op = randn(N, N)
    op += op'

    function tfunc(A, b)
        x = BackwardsLinalg.lstsq(A, b)
        return x'*op*x
    end
    g1 = Zygote.gradient(tfunc, A, b)
    backend = DifferentiationInterface.AutoMooncake(; config=nothing)
    wrapped(x) = tfunc(x...)
    Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(BackwardsLinalg.lstsq), Matrix{Float64}, Vector{Float64}}
    prep = DifferentiationInterface.prepare_gradient(wrapped, backend, (A, b))
    g2 = DifferentiationInterface.gradient(wrapped, prep, backend, (A, b))
    @test all(g1 .≈ g2)
end
