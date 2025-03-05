using BackwardsLinalg
using Test, Random, LinearAlgebra
using Zygote

function gradient_check(f, args...; η = 1e-5)
    g = gradient(f, args...)
    dy_expect = η*sum(abs2.(g[1]))
    @show dy_expect
    dy = f(args...)-f([gi === nothing ? arg : arg.-η.*gi for (arg, gi) in zip(args, g)]...)
    @show dy
    isapprox(dy, dy_expect, rtol=1e-2, atol=1e-8)
end


@testset "symeigen for hermite" begin
    T = ComplexF64
    M =10
    A = randn(T,M,M)
    A += A'
    function tfunc(A)
        E,U = BackwardsLinalg.symeigen(A)
        return sum(abs2.(E)) + sum(abs2.(U[:,1]))
    end

    @test gradient_check(tfunc,A)
end

@testset "symeigen for normal" begin
    Random.seed!(6)
    T = ComplexF64
    M = 4
    A = randn(T,M,M)
    Q = LinearAlgebra.qr(A).Q
    S = diagm(randn(T,M))
    A =Q*S*Q'
    function tfunc(A)
        E,U = BackwardsLinalg.symeigen(A)
        return sum(abs2.(E)) + sum(abs2.(U[:,1]))
    end

    @test gradient_check(tfunc,A)
end