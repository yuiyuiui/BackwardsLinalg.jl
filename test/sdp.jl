using JuMP, SCS, LinearAlgebra, Random
using Test, Zygote, BackwardsLinalg


function gradient_check(f, args...; η = 1e-5)
    g = gradient(f, args...)
    dy_expect = η*sum(abs2.(g[1]))
    @show dy_expect
    dy = f(args...)-f([gi === nothing ? arg : arg.-η.*gi for (arg, gi) in zip(args, g)]...)
    @show dy
    isapprox(dy, dy_expect, rtol=1e-2, atol=1e-8)
end


# 定义数据
Random.seed!(123)  # 设置随机种子
n = 4  # 矩阵的维度

# 生成对称的目标矩阵 C
C = rand(n, n)
C = (C + C') / 2  # 确保对称性

# 生成对称的约束矩阵 A1 和 A2
A1 = rand(n, n)
A1 = (A1 + A1') / 2  # 确保对称性

A2 = rand(n, n)
A2 = (A2 + A2') / 2  # 确保对称性

# 生成约束的右侧值 b1 和 b2
b1 = rand()  # 约束 1 的右侧值 b1，确保可行
b2 = rand()  # 约束 2 的右侧值 b2，确保可行

# 使用 JuMP + SCS 求解
model = Model(SCS.Optimizer)
@variable(model, X[1:n, 1:n], PSD)  # 定义半正定矩阵 X
@objective(model, Min, tr(C * X))   # 目标是最小化 tr(C * X)
@constraint(model, tr(A1 * X) == b1)  # 约束 1: tr(A1 * X) = b1
@constraint(model, tr(A2 * X) == b2)  # 约束 2: tr(A2 * X) = b2
optimize!(model)  # 求解问题

# 检查求解状态并输出结果
if termination_status(model) == MOI.OPTIMAL
    println("JuMP + SCS 结果：")
    println("目标函数值: ", objective_value(model))
    println("最优解 X:")
    println(value.(X))
else
    println("JuMP + SCS 求解失败")
    println("求解状态: ", termination_status(model))
end
E,U = eigen(value.(X))



