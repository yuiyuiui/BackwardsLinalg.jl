function rrule(::typeof(qr), A)
	Q, R = qr(A)
	function pullback(dy)
		ΔA = @thunk qr_back(A, Q, R, unthunk.(dy)...)
		return (NoTangent(), ΔA)
	end
	return (Q, R), pullback
end

function rrule(::typeof(qr), A::AbstractMatrix, pivot::Val{true})
	Q, R, P = qr(A, pivot)
	function pullback(dy)
		ΔA = @thunk qr_back(Q * R, Q, R, unthunk(dy[1]), unthunk(dy[2])) * P'
		return (NoTangent(), ΔA, NoTangent())
	end
	return (Q, R, P), pullback
end

function rrule(::typeof(lq), A)
	L, Q = lq(A)
	function pullback(dy)
		ΔA = @thunk lq_back(A, L, Q, unthunk.(dy)...)
		return (NoTangent(), ΔA)
	end
	return (L, Q), pullback
end

function rrule(::typeof(svd), A)
	U, S, V = svd(A)
	@info "svd forward" U S V
	function pullback(dy)
		@info "svd pullback"
		ΔA = @thunk svd_back(U, S, V, unthunk.(dy)...)
		return (NoTangent(), ΔA)
	end
	return (U, S, V), pullback
end

function rrule(::typeof(rsvd), A, args...; kwargs...)
	U, S, V = rsvd(A, args...; kwargs...)
	function pullback(dy)
		ΔA = @thunk svd_back(U, S, V, unthunk.(dy)...)
		return (NoTangent(), ΔA)
	end
	return (U, S, V), pullback
end

function rrule(::typeof(symeigen), A)
	E, U = symeigen(A)
	function pullback(dy)
		ΔA = @thunk symeigen_back(E, U, unthunk.(dy)...)
		return (NoTangent(), ΔA)
	end
	return (E, U), pullback
end

function rrule(::typeof(normeigen), A)
	E, U = normeigen(A)
	function pullback(dy)
		ΔA = @thunk normeigen_back(E, U, unthunk.(dy)...)
		return (NoTangent(), ΔA)
	end
	return (E, U), pullback
end

function rrule(::typeof(arg_lstsq), A, b)
	x = arg_lstsq(A, b)
	function pullback(dy)
		Δy = unthunk(dy)
		ΔA, Δb = @thunk arg_lstsq_back(A, b, x, Δy)
		return (NoTangent(), ΔA, Δb)
	end
	return x, pullback
end

function rrule(::typeof(lstsq), A, b)
	a = lstsq(A, b)
	function pullback(ā)
		Ā, b̄ = @thunk lstsq_back(A, b, unthunk(ā))
		return (NoTangent(), Ā, b̄)
	end
	return a, pullback
end

function rrule(::typeof(mxmul), A, B)
	C = mxmul(A, B)
	function pullback(dy)
		Ā, B̄ = @thunk mxmul_back(A, B, unthunk(dy))
		return (NoTangent(), Ā, B̄)
	end
	return C, pullback
end

function rrule(::typeof(scha_norm), A, p)
	a = scha_norm(A, p)
	function pullback(ā)
		Ā = @thunk scha_norm_back(A, p, unthunk(ā))
		return (NoTangent(), Ā, NoTangent())
	end
	return a, pullback
end

function rrule(::typeof(cls), A)
	L = cls(A)
	function pullback(L̄)
		Ā = @thunk cls_back(A, unthunk(L̄))
		return (NoTangent(), Ā)
	end
	return L, pullback
end

function rrule(::typeof(det), A)
	a = det(A)
	function pullback(ā)
		Ā = @thunk det_back(A, unthunk(ā))
		return (NoTangent(), Ā)
	end

	return a, pullback
end

function rrule(::typeof(inv), A)
	B = inv(A)
	function pullback(B̄)
		Ā = @thunk inv_back(A, unthunk(B̄))
		return (NoTangent(), Ā)
	end

	return B, pullback
end

function rrule(::typeof(lneq), A, b)
	x = lneq(A, b)
	function pullback(dy)
		Δy = unthunk(dy)
		ΔA, Δb = @thunk lneq_back(A, b, x, Δy)
		return (NoTangent(), ΔA, Δb)
	end
	return x, pullback
end

function rrule(::typeof(lu), A)
	x = lu(A)
	function pullback(dy)
		Ā = @thunk lu_back(A, unthunk.(dy)...)
		return (NoTangent(), Ā)
	end
	return x, pullback
end

function rrule(::typeof(norm_anlfunc), f, df, A)
	B = norm_anlfunc(f, df, A)
	function pullback(B̄)
		Ā = @thunk norm_anlfunc_back(f, df, A, unthunk(B̄))
		return (NoTangent(), NoTangent(), NoTangent(), Ā)
	end
	return B, pullback
end

function rrule(::typeof(lp), c, A, b)
	x, a = lp(c, A, b)
	function pullback(ȳ)
		c̄, Ā, b̄ = @thunk lp_back(c, A, b, x, unthunk.(ȳ)...)
		return (NoTangent(), c̄, Ā, b̄)
	end
	return (x, a), pullback
end

function rrule(::typeof(gmres), A, b; args...)
	x = gmres(A, b; args...)
	function pulllback(x̄)
		Ā, b̄ = @thunk gmres_back(A, b, unthunk(x̄); args...)
		return (NoTangent(), Ā, b̄)
	end
	return x, pulllback
end


function rrule(::typeof(pf), A)
	pfA = pf(A)
	function pulllback(ā)
		Ā = @thunk pf_back(A, pfA, unthunk(ā))
		return (NoTangent(), Ā)
	end
	return pfA, pulllback
end

function rrule(::typeof(sdp), C, A, b)
	X = sdp(C, A, b)
	function pullback(X̄)
		C̄, Ā, b̄ = @thunk sdp_back(C, A, b, X, unthunk(X̄))
		return (NoTangent(), C̄, Ā, b̄)
	end
	return X, pullback
end