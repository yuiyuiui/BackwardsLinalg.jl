function lstsq(A::Matrix{T},b::Vector{T}) where T<:Number
    A1=A'*A
    @assert LinearAlgebra.det(A1)!=0
    return A1\(A'*b)
end

function lstsq_back(A::Matrix{T},b::Vector{T},x,x̄) where T
    Q,R = LinearAlgebra.qr(A)
    b̄ = Q*(R')^(-1)*x̄
    Ā = (b-A*x)*x̄'*(R'*R)^(-1) -Q*(R')^(-1)*x̄*x'
    return Ā,b̄
end


