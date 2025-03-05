using JuMP, SCS, LinearAlgebra, Random

# 定义数据
Random.seed!(3)
n = 2  # 矩阵的维度
C = exp.(rand(2,2))  # 目标矩阵 C
A1 = exp.(rand(2,2)) # 约束矩阵 A1
A2 = exp.(rand(2,2)) # 约束矩阵 A2
C += C'
A1 += A1' 
A2 += A2'
b1 = exp(rand())          # 约束 1 的右侧值 b1
b2 = exp(rand())          # 约束 2 的右侧值 b2

# 使用 JuMP + SCS 求解
model = Model(SCS.Optimizer)
@variable(model, X[1:n, 1:n], PSD)
@objective(model, Min, tr(C * X))
@constraint(model, tr(A1 * X) == b1)
@constraint(model, tr(A2 * X) == b2)
optimize!(model)

if termination_status(model) == MOI.OPTIMAL
    println("JuMP + SCS 结果：")
    println("目标函数值: ", objective_value(model))
    println("最优解 X:")
    println(value.(X))
else
    println("JuMP + SCS 求解失败")
end



# =======================

using LinearAlgebra

function solve_sdp(C, A_list, b_list; max_iter=1000, step_size=0.01, tol=1e-6)
    """
    使用投影梯度法求解标准形式 SDP：
        min Tr(C * X)
        s.t. Tr(A_i * X) = b_i, for all i
             X ⪰ 0

    输入:
        C: 目标矩阵 (n x n 对称矩阵)
        A_list: 约束矩阵列表 (每个元素为 n x n 对称矩阵)
        b_list: 约束右侧值列表 (每个元素为标量)
        max_iter: 最大迭代次数 (默认 1000)
        step_size: 步长 (默认 0.01)
        tol: 收敛容忍度 (默认 1e-6)

    输出:
        min_value: 最小值
        X_opt: 最优解 X (n x n 半正定矩阵)
        iter: 实际迭代次数
    """
    n = size(C, 1)  # 矩阵维度
    m = length(A_list)  # 约束个数

    # 初始化变量 X
    X = zeros(n, n)  # 初始点为零矩阵

    # 投影梯度法主循环
    for iter in 1:max_iter
        # 计算梯度 ∇f(X) = C
        grad = C

        # 更新 X：X = X - step_size * grad
        X_new = X - step_size * grad

        # 投影到可行域：满足约束 Tr(A_i * X) = b_i
        # 使用拉格朗日乘子法修正 X_new
        # 构建线性方程组：M * λ = v
        M = zeros(m, m)  # M[i, j] = Tr(A_i * A_j)
        v = zeros(m)     # v[i] = Tr(A_i * X_new) - b_list[i]

        for i in 1:m
            for j in 1:m
                M[i, j] = tr(A_list[i] * A_list[j])
            end
            v[i] = tr(A_list[i] * X_new) - b_list[i]
        end

        # 解线性方程组 M * λ = v
        λ = M \ v

        # 更新 X_new
        for i in 1:m
            X_new = X_new - λ[i] * A_list[i]
        end

        # 投影到半正定锥：将 X_new 的特征值截断为非负
        F = eigen(Symmetric(X_new))  # 使用对称矩阵确保数值稳定性
        X_new = F.vectors * Diagonal(max.(F.values, 0)) * F.vectors'

        # 检查收敛条件
        if norm(X_new - X) < tol
            println("收敛于第 ", iter, " 次迭代")
            return tr(C * X_new), X_new, iter
        end

        # 更新 X
        X = X_new
    end

    println("达到最大迭代次数 ", max_iter)
    return tr(C * X), X, max_iter
end

# 使用改进后的 solve_sdp 函数求解
A_list = [A1, A2]
b_list = [b1, b2]
min_value, X_opt, iter = solve_sdp(C, A_list, b_list, max_iter=1000, step_size=0.01, tol=1e-6)

println("\n改进后的 solve_sdp 结果：")
println("目标函数值: ", min_value)
println("最优解 X:")
println(X_opt)
println("迭代次数: ", iter)

nextfloat(1.0)-1.0 == eps(Float64)