#import "@preview/cetz:0.2.2": *
#import "@preview/unequivocal-ams:0.1.2": ams-article, theorem, proof
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm
#show link: set text(blue)

#let jinguo(txt) = {
  text(blue, [[JG: #txt]])
}

#set math.equation(numbering: "(1)")

#show: ams-article.with(
  title: [A technical note on automatic differentiation],
  // authors: (
  //   (
  //     name: "Yi-Dai Zhang",
  //     department: [Advanced Materials Thrust],
  //     organization: [Hong Kong University of Science and Technology (Guangzhou)],
  //   ),
  //   (
  //     name: "Lei Wang",
  //     organization: [Institute of Physics, Chinese Academy of Sciences],
  //   ),
  //   (
  //     name: "Jin-Guo Liu",
  //     department: [Advanced Materials Thrust],
  //     organization: [Hong Kong University of Science and Technology (Guangzhou)],
  //     email: "jinguoliu@hkust-gz.edu.cn",
  //   ),
  // ),
  abstract: [Automatic differentiation (AD) is a technique to compute the derivative of a function represented by a computational process. It is widely used in physics simulations, machine learning, optimization, and other fields. In this review, we focus on the application of AD in physics simulations.],
  bibliography: bibliography("refs.bib"),
)

// The ASM template also provides a theorem function.
#let definition(title, body, numbered: true) = figure(
  body,
  kind: "theorem",
  supplement: [Definition (#title)],
  numbering: if numbered { "1" },
)
#let rulebox(title, rule) = block(width: 100%, stroke: black, radius: 4pt, inset: 10pt)[
_Function_: #title\
\
_Backward rule_: #rule
]


#set math.equation(numbering: "(1)")

= Introduction <introduction>

The automatic differentiation (AD) is a technique to compute the derivative of a function represented by a computational process.
It can be classified into two categories: forward mode and reverse mode@Li2017 @Griewank2008.
_Forward mode AD_ presumes the scalar input.
Given a program with scalar input $t$, we can denote the intermediate variables of the program as $bold(y)_i$, and their _derivatives_ as $dot(bold(y)_i) = (partial bold(y)_i)/(partial t)$.
The _forward rule_ defines the transition between $bold(y)_i$ and $bold(y)_(i+1)$
$
dot(bold(y))_(i+1) = (diff bold(y)_(i+1))/(diff bold(y)_i) dot(bold(y))_i.
$
// In the program, we can define a *dual number* with two fields, just like a complex number.
In an automatic differentiation engine, the Jacobian matrix $(diff bold(y)_(i+1))/(diff bold(y)_i)$ is almost never computed explicitly in memory as it can be costly.
Instead, the forward mode automatic differentiation can be implemented by overloading the function $f_i$ as
$ f_i^("forward"): (bold(y)_i, dot(bold(y))_i) arrow.bar (bold(y)_(i+1), (diff bold(y)_(i+1))/(diff bold(y)_i) dot(bold(y))_i), $
which updates both the value and the derivative of the intermediate variables.
When we have multiple inputs, the forward mode AD have to repeatedly evaluate the derivatives for each input, which is computationally expensive.

//Let us consider a computational process that computes the value of a function $bold(y) = f(bold(x))$.
To circumvent this issue, the _reverse mode AD_ is proposed, which presumes a scalar output $cal(L)$, or the loss function.
Given a program with scalar output $cal(L)$, we can denote the intermediate variables of the program as $bold(y)_i$, and their _adjoints_ as $overline(bold(y))_i = (partial cal(L))/(partial bold(y)_i)$.
The _backward rule_ defines the transition between $overline(bold(y))_(i+1)$ and $overline(bold(y))_i$
$
overline(bold(y))_i = overline(bold(y))_(i+1) (partial bold(y)_(i+1))/(partial bold(y)_i).
$
Again, in the program, there is no need to compute the Jacobian matrix explicitly in memory.
We define the backward function $overline(f)_i$ as
$ overline(f)_i: ("TAPE", overline(bold(y))_(i+1)) arrow.bar ("TAPE", overline(bold(y))_(i+1) (partial bold(y)_(i+1))/(partial bold(y)_i)), $
where "TAPE" is a cache for storing the intermediate variables that required for implementing the backward rule.
Due to the "TAPE", the reverse mode AD is much harder to implement than the forward mode AD.
The forward mode AD has a natural order of visiting the intermediate variables, which can be supported by running the program forwardly.
While the reverse mode AD has to visit the intermediate variables in the reversed order, we have to run the program forwardly and store the intermediate variables in a stack called "TAPE".
Then in the backward pass, we pop the intermediate variables from the "TAPE" and compute the adjoint of the variables.

As shown in @fig:computational_graph, the computational process can be represented as a directed acyclic graph
(DAG) where nodes are operations and edges are data dependencies.
The forward pass computes the value of the function and stores the intermediate variables in the "TAPE".
The backward pass pops the intermediate variables from the "TAPE" and computes the adjoint of the variables.
#jinguo([TODO: polish the figure])

#figure((
  canvas({
    import draw: *
    let s(x) = text(8pt, x)
    for (x, y, txt, nm, st) in ((-0.2, 0.5, s[$id$], "t", black), (1, 0, s[$cos$], "cos(t)", black), (1, 1, s[$sin$], "sin(t)", black), (2.5, 0, [$*$], "*", black)) {
      circle((x, y), radius: 0.3, name: nm, stroke: st)
      content((x, y), txt)
    }
    line((rel: (-1, 0), to: "t"), "t", name: "l0")
    line("t", "cos(t)", name: "l1")
    line("t", "sin(t)", name: "l2")
    line("cos(t)", "*", name: "l3")
    line("sin(t)", "*", name: "l4")
    line((rel: (-1, -1), to: "*"), "*", name: "l5")
    line("*", (rel: (1, 0), to: "*"), name: "l6")
    mark("l0.start", "l0.mid", end: "straight")
    mark("l1.start", "l1.mid", end: "straight")
    mark("l2.start", "l2.mid", end: "straight")
    mark("l3.start", "l3.mid", end: "straight")
    mark("l4.start", "l4.mid", end: "straight")
    mark("l5.start", "l5.mid", end: "straight")
    mark("l6.start", "l6.mid", end: "straight")
    content((rel: (0, 0.2), to: "l0.mid"), s[$theta$])
    content((rel: (0, -0.2), to: "l1.mid"), s[$theta$])
    content((rel: (0, 0.2), to: "l2.mid"), s[$theta$])
    content((rel: (0, -0.2), to: "l3.mid"), s[$cos theta$])
    content((rel: (0.2, 0.2), to: "l4.mid"), s[$sin theta$])
    content((rel: (-0.2, -0.2), to: "l6.end"), s[$y$])
    content((rel: (0.1, -0.1), to: "l5.mid"), s[$r$])

    content((1, -1.5), [Forward Pass])

    set-origin((6, 0))
    for (x, y, txt, nm, st) in ((-0.2, 0.5, s[$id$], "t", black), (1, 0, s[$cos$], "cos(t)", black), (1, 1, s[$sin$], "sin(t)", black), (2.5, 0, [$*$], "*", black)) {
      circle((x, y), radius: 0.3, name: nm, stroke: st)
      content((x, y), txt)
    }
    line((rel: (-1, 0), to: "t"), "t", name: "l0")
    line("t", "cos(t)", name: "l1")
    line("t", "sin(t)", name: "l2")
    line("cos(t)", "*", name: "l3")
    line("sin(t)", "*", name: "l4")
    line((rel: (-1, -1), to: "*"), "*", name: "l5")
    line("*", (rel: (1, 0), to: "*"), name: "l6")
    mark("l0.end", "l0.mid", end: "straight")
    mark("l1.end", "l1.mid", end: "straight")
    mark("l2.end", "l2.mid", end: "straight")
    mark("l3.end", "l3.mid", end: "straight")
    mark("l4.end", "l4.mid", end: "straight")
    mark("l5.end", "l5.mid", end: "straight")
    mark("l6.end", "l6.mid", end: "straight")
    content((rel: (-0.7, 0.2), to: "l0.mid"), s[$r (sin^2 theta + cos^2 theta)$])
    content((rel: (-0.3, -0.2), to: "l1.mid"), s[$r sin^2 theta$])
    content((rel: (-0.3, 0.2), to: "l2.mid"), s[$r cos^2 theta$])
    content((rel: (0, -0.2), to: "l3.mid"), s[$r sin theta$])
    content((rel: (0.3, 0.2), to: "l4.mid"), s[$r cos theta$])
    content((rel: (-0.2, -0.2), to: "l6.end"), s[$1$])
    content((rel: (0.6, -0.1), to: "l5.mid"), s[$sin theta cos theta$])

    content((1, -1.5), [Backward Pass])
  })
), caption: [The computational graph for calculating $y = r cos theta sin theta$. Nodes are operations and edges are variables.
The node "$id$" is the copy operation.]) <fig:computational_graph>

== Obtaining Hessian

The second order gradient, or Hessian, can be computed by taking the Jacobian of the gradient.
Note that the program to compute the gradient of a function is also a differentiable program.
Consider a multivariate function $f: bb(R)^n arrow.r bb(R)$, the gradient function $nabla f: bb(R)^n arrow.r bb(R)^n$ is also a differentiable function.
After computing the gradient with the reverse mode AD, we can use the forward mode AD to compute the Hessian.
The reason why we can use the forward mode AD to compute the Hessian is that the gradient function $nabla f$ has equal number of input and output dimensions.
The forward mode AD is more memory efficient than the reverse mode AD in this case.

== Complex valued automatic differentiation <complex-valued-automatic-differentiation>
Complex valued AD considers the problem that a function takes complex variables as inputs, while the loss is still real valued.
Since such function cannot be holomorphic, or complex differentiable, the adjoint of a such a function is defined by treating the real and imaginary parts of the input as independent variables.
Let $z = x + i y$ be a complex variable, and $cal(L)$ be a real loss function.
The adjoint of $z$ is defined as
$
  overline(z) = overline(x) + i overline(y).
$
If we change $z$ by a small amount $delta z = delta x + i delta y$, the loss function $cal(L)$ will change by
$ delta cal(L) = (overline(z)^* delta z + h.c.)\/2 = overline(x) delta x + overline(y) delta y. $

= Differentiating linear algebra operations <differentiating-linear-algebra-operations>


== Notations

We derived the following useful relations:
$ tr[A(C compose B)] = sum A^T compose C compose B = tr((C compose A^T)^T B) = tr(C^T compose A)B $ <eq:tr_compose>

$ (C compose A)^T = C^T compose A^T $ <eq:transpose_compose>

Let $cal(L)$ be a real function of a complex variable $x$, $ (diff cal(L))/(diff x^*) = ((diff cal(L))/(diff x))^* $ <eq:diff_complex>



== Matrix multiplication <matrix-multiplication>

#rulebox([Matrix multiplication $C = A B$, where $A in CC^(m times n)$ and $B in CC^(n times p)$.],
[
  $ cases(
    overline(A) &= overline(C) B^dagger,
    overline(B) &= A^dagger overline(C)
  ) $
])


// === Matrix multiplication
// Let $cal(T)$ be a stack, and $x arrow.r cal(T)$ and $x arrow.l cal(T)$ be the operation of pushing and poping an element from this stack.
// Given $A in R^(l times m)$ and $B in R^(m times n)$, the forward pass computation of matrix multiplication is
// $ 
// cases(
//   C = A B,
//   A arrow.r cal(T),
//   B arrow.r cal(T),
//   dots
// )
// $

// Let the adjoint of $x$ be $overline(x) = (partial cal(L))/(partial x)$, where $cal(L)$ is a real loss as the final output.
// The backward pass computes
// $ 
// cases(
//   dots,
//   B arrow.l cal(T),
//   overline(A) = overline(C)B,
//   A arrow.l cal(T),
//   overline(B) = A overline(C)
// )
// $

// The rules to compute $overline(A)$ and $overline(B)$ are called the backward rules for matrix multiplication. They are crucial for rule based automatic differentiation.

Let us introduce a small perturbation $delta A$ on $A$ and $delta B$ on $B$,

$ delta C = delta A B + A delta B $

$ delta cal(L) = tr(delta C^T overline(C)) = 
tr(delta A^T overline(A)) + tr(delta B^T overline(B)) $

It is easy to see
$ delta L = tr((delta A B)^T overline(C)) + tr((A delta B)^T overline(C)) = 
tr(delta A^T overline(A)) + tr(delta B^T overline(B)) $

We have the backward rules for matrix multiplication as
$ 
cases(
  overline(A) = overline(C)B^T,
  overline(B) = A^T overline(C)
)
$


== Tensor network contraction <tensor-network-contraction>

#rulebox([
Tensor network contraction
$ O_(sigma_i) = "contract"(Lambda, cal(T), sigma_o), $
where $Lambda$ is a set of variables, $cal(T) = {T_(sigma_1), T_(sigma_2), ..., T_(sigma_m)}$ is a set of input tensors, and $sigma_o$ is a set of output variables.
],
[
$ overline(T)_(sigma_i) = ("contract"(Lambda, cal(T) without {T_(sigma_i)} union {overline(O)^*_(sigma_o)}, sigma_i))^* $ <eq:einback>
])

In this section, we will derive @eq:einback, which is the backward rule for a pairwise tensor contraction, denoted by $"contract"(Lambda, {A_(V_a), B_(V_b)}, V_c)$.
Let $cal(L)$ be a loss function of interest, where its differential form is given by:

$
  delta cal(L) &= "contract"(V_a, {delta A_(V_a), overline(A)_(V_a)}, nothing) + "contract"(V_b, {delta B_(V_b), overline(B)_(V_b)}, nothing)\
               &= "contract"(V_c, {delta C_(V_c), overline(C)_(V_c)}, nothing)
$ <eq:diffeq>

The goal is to find $overline(A)_(V_a)$ and $overline(B)_(V_b)$ given $overline(C)_(V_c)$.
This can be achieved by using the differential form of tensor contraction, which states that:

$
  delta C = "contract"(Lambda, {delta A_(V_a), B_(V_b)}, V_c) + "contract"(Lambda, {A_(V_a), delta B_(V_b)}, V_c)
$

By inserting this result into @eq:diffeq, we obtain:

$
  delta cal(L) &= "contract"(V_a, {delta A_(V_a), overline(A)_(V_a)}, nothing) + "contract"(V_b, {delta B_(V_b), overline(B)_(V_b)}, nothing)\
               &= "contract"(Lambda, {delta A_(V_a), B_(V_b), overline(C)_(V_c)}, nothing) + "contract"(Lambda, {A_(V_a), delta B_(V_b), overline(C)_(V_c)}, nothing)
$

Since $delta A_(V_a)$ and $delta B_(V_b)$ are arbitrary, the above equation immediately implies @eq:einback.

== The least square problem <least-square-problem>
#jinguo([complex valued version needs to be added.])
#rulebox([
The real valued least square problem in the matrix form:
$
min_x ||A x - b||^2,
$
where $A in bb(R)^(m times n)$ and $b in bb(R)^m$ with $m > n$ are inputs, $x$ is the output.
],
[
$
&overline(b) = Q R(R^T R)^(-1) overline(x) = Q (R^T)^(-1) overline(x)\
&overline(A) = (b - A x)overline(x)^T R^(-1)(R^T)^(-1) -   Q(R^T)^(-1) overline(x) x^T
$
])

The solution of the least square problem is given by:
$
x = (A^T A)^(-1) A^T b quad "or" quad (A^T A)x = A^T b.
$ <eq:lsq_sol>
Note that this defining equation is usually not how we compute the solution. In practice, we use the QR decomposition to compute the solution.

Let us denote the adjoint of a variable $v$ as $overline(v) "s.t." delta cal(L) = overline(v) delta v$, where $cal(L)$ is a hypothetical loss function.
Since we have the mapping $(A, b) arrow.r x$, we have the following differential relation:
$
  delta cal(L) = tr(overline(x)^T delta x) = tr(overline(A)^T delta A) + tr(overline(b)^T delta b).
$ <eq:lsq_diff>
The *goal* is to find $overline(A)$ and $overline(b)$ given $overline(x)$.

By considering @eq:lsq_sol, we also have:
$
(A^T + delta A^T) (A + delta A) (x + delta x) = (A^T + delta A^T) (b + delta b).
$
Keeping only the first order terms, we have:
$
&delta A^T A x + A^T delta A x + A^T A delta x = A^T delta b + delta A^T b\
arrow.double.r &delta x = (A^T A)^(-1) (A^T delta b + delta A^T b - delta A^T A x - A^T delta A x).
$
Inserting the above into the differential relation @eq:lsq_diff, we have:
$
  &tr(overline(x)^T (A^T A)^(-1) (A^T delta b + delta A^T b - delta A^T A x - A^T delta A x)) = tr(overline(A)^T delta A) + tr(overline(b)^T delta b)\
  = &tr(overline(x)^T (A^T A)^(-1)A^T delta b) + tr(overline(x)^T (A^T A)^(-1) delta A^T (b - A x) - overline(x)^T (A^T A)^(-1) A^T delta A x)\
  = &tr(overline(x)^T (A^T A)^(-1)A^T delta b) + tr((b - A x)^T delta A (A^T A)^(-1) overline(x) - overline(x)^T (A^T A)^(-1) A^T delta A x)\
  = &tr(overline(x)^T (A^T A)^(-1)A^T delta b) + tr((A^T A)^(-1)overline(x)(b - A x)^T delta A  - x overline(x)^T (A^T A)^(-1) A^T delta A)
$
where we have used the following relations
- $tr(A B C) = tr(B C A) = tr(C A B)$
- $tr(X) = tr(X^T)$

Since $delta b$ and $delta A$ are arbitrary, we have:
$
&overline(b) = A (A^T A)^(-1) overline(x)\
&overline(A) = (b - A x)overline(x)^T (A^T A)^(-1) -   A (A^T A)^(-1) overline(x) x^T
$

Let $A = Q R$ be the QR decomposition of $A$, where $Q in bb(R)^(m times n)$ is an orthogonal matrix ($Q^T Q = bb(I)$) and $R in bb(R)^(n times n)$ is an *invertible* upper triangular matrix. We have:
$
&overline(b) = Q R(R^T R)^(-1) overline(x) = Q (R^T)^(-1) overline(x)\
&overline(A) = (b - A x)overline(x)^T R^(-1)(R^T)^(-1) -   Q(R^T)^(-1) overline(x) x^T
$

=== How to compute the adjoint
From computational perspective, we
1. obtain $y = (R^T)^(-1) overline(x)$ by solving the linear system $R^T y = overline(x)$, then we have:
  $
  &overline(b) = Q y\
  &overline(A) = (b - A x)y^T (R^T)^(-1) -   overline(b) x^T
  $
2. obtain $z = (R)^(-1) y$ by solving the linear system $R z = y$, then we have:
  $
  &overline(A) = (b - A x)z^T -   overline(b) x^T
  $

== QR decomposition <qr-decomposition>
#jinguo([with pivoting? thin and wide QR?])

#rulebox([QR decomposition.
Let $A$ be a full rank matrix, the QR decomposition is defined as
$ A = Q R $
with $Q^dagger Q = bb(I)$, so that $d Q^dagger Q + Q^dagger d Q = 0$. $R$ is a complex upper triangular matrix, with diagonal part real.
],
[
$
  overline(A) = overline(Q) + Q "copyltu"(M)R^(-dagger),
$
where $M = R^(-1)overline(R)^dagger - overline(Q)^dagger Q$.
The $"copyltu"$ takes conjugate when copying elements to upper triangular part.


])

The backward rules for QR decomposition are derived in multiple references, including @Hubig2019 and @Liao2019. To derive the backward rules, we first consider differentiating the QR decomposition
@Seeger2017, @Liao2019

$ d A = d Q R + Q d R $

$ d Q = d A R^(-1) - Q d R R^(-1) $

$ cases(
  Q^dagger d Q = d C - d R R^(-1),
  d Q^dagger Q = d C^dagger - R^(-dagger)d R^dagger
) $

where $d C = Q^dagger d A R^(-1)$.

Then

$ d C + d C^dagger = d R R^(-1) + (d R R^(-1))^dagger $

Notice $d R$ is upper triangular and its diag is lower triangular, this restriction gives

$ U compose (d C + d C^dagger) = d R R^(-1) $

where $U$ is a mask operator that its element value is $1$ for upper triangular part, $0.5$ for diagonal part and $0$ for lower triangular part. One should also notice here both $R$ and $d R$ has real diagonal parts, as well as the product $d R R^(-1)$.

We have

$ 
  d cal(L) &= tr[overline(Q)^dagger d Q + overline(R)^dagger d R + "h.c."],\
  &= tr[overline(Q)^dagger d A R^(-1) - overline(Q)^dagger Q d R R^(-1) + overline(R)^dagger d R + "h.c."],\
  &= tr[R^(-1)overline(Q)^dagger d A + R^(-1)(-overline(Q)^dagger Q + R overline(R)^dagger)d R + "h.c."],\
  &= tr[R^(-1)overline(Q)^dagger d A + R^(-1)M d R + "h.c."]
$

here, $M = R overline(R)^dagger - overline(Q)^dagger Q$. Plug in $d R$ we have

$ 
  d cal(L) &= tr[R^(-1)overline(Q)^dagger d A + M[U compose (d C + d C^dagger)] + "h.c."],\
  &= tr[R^(-1)overline(Q)^dagger d A + (M compose L)(d C + d C^dagger) + "h.c."] #h(2em),\
  &= tr[(R^(-1)overline(Q)^dagger d A + "h.c.") + (M compose L)(d C + d C^dagger) + (M compose L)^dagger (d C + d C^dagger)],\
  &= tr[R^(-1)overline(Q)^dagger d A + (M compose L + "h.c.")d C + "h.c."],\
  &= tr[R^(-1)overline(Q)^dagger d A + (M compose L + "h.c.")Q^dagger d A R^(-1)] + "h.c."
$

where $L = U^dagger = 1-U$ is the mask of lower triangular part of a matrix.
In the second line, we have used @eq:tr_compose.

$
  overline(A)^dagger &= R^(-1)[overline(Q)^dagger + (M compose L + "h.c.")Q^dagger],\
  overline(A) &= [overline(Q) + Q "copyltu"(M)]R^(-dagger),\
  &= [overline(Q) + Q "copyltu"(M)]R^(-dagger)
$

Here, the $"copyltu"$ takes conjugate when copying elements to upper triangular part.

== Eigenvalue decomposition <eigenvalue-decomposition>

#rulebox([
Symmetric eigenvalue decomposition
$ A = U E U^dagger, $
where the input $A$ is a Hermitian matrix, the outputs $U$ is a unitary matrix and $E$ is a diagonal matrix.
],
[
$
overline(A) = U[overline(E) + 1/2(overline(U)^dagger U compose F + "h.c.")]U^dagger
$
where $F_(i j)=(E_j - E_i)^(-1)$.
])

#jinguo([To be added])

== Singular value decomposition <singular-value-decomposition>

- SVD @Hubig2019, @Townsend2016, @Giles2008
- Complex SVD @Wan2019
- Truncated SVD @Francuz2023

#rulebox([
Complex valued singular value decomposition
$
&A = U S V^dagger,\ &V^dagger V = I,\ &U^dagger U = I,\ &S = "diag"(s_1, ..., s_n),
$ <eq:svd>
where the input $A$ is a complex matrix, the outputs $U$ is a unitary matrix, $S$ is a real diagonal matrix and $V$ is a unitary matrix. We also apply an extra constraint that the loss function $cal(L)$ is real and is invariant under the gauge transformation: $U arrow.r U Lambda$, $V arrow.r V Lambda$, where $Lambda$ is defined as $"diag"(e^(i phi_1), ..., e^(i phi_n))$.
],
[
$
  overline(A) = &U(J + J^dagger) S V^dagger + (I-U U^dagger)overline(U)S^(-1)V^dagger,\
  &+ U S(K + K^dagger)V^dagger + U S^(-1) overline(V)^dagger (I - V V^dagger),\
  &+ U (overline(S) compose I) V^dagger,\
  &+ 1/2 U (S^(-1) compose(U^dagger overline(U))-h.c.)V^dagger
$ <eq:svd_loss_diff_full>
where $J=F compose(U^dagger overline(U))$, $K=F compose(V^dagger overline(V))$ and $F_(i j) = cases( 1/(s_j^2-s_i^2) \, &i!=j, 0\, &i=j)$.
])

We start with the following two relation
$
  2 delta cal(L) = tr[overline(A)^dagger delta A + h.c.] = tr[overline(U)^dagger delta U + overline(V)^dagger delta V + h.c.] + 2tr[overline(S) delta S]
$ <eq:loss_diff>
//where we have used @eq:diff_complex.

$
delta A = delta U S V^dagger + U delta S V^dagger + U S delta V^dagger
$ <eq:svd_diff>
//The clue is to resolve the right hand side of @eq:loss_diff into the form of $tr[f(A, overline(U), overline(V), overline(S)) delta A]$, then we will have $overline(A) = f(A, overline(U), overline(V), overline(S))^dagger$ as $delta A$ is arbitrary.

We first sandwich @eq:svd_diff between $U^dagger$ and $V$ and obtain
$
U^dagger delta A V &= U^dagger delta U S + delta S + S delta V^dagger V.
$
Then we denote $delta C=U^dagger delta U$, $delta D = delta V^dagger V$ and $delta P = U^dagger delta A V$,
then by using the second and third line in @eq:svd, we have $d U$ and $d V$ are skew-symmetric, i.e.

$ cases(
  delta C^dagger + delta C = 0,
  delta D^dagger + delta D = 0
) $ <eq:svd_delta_c_d>

We can simplify @eq:svd_diff as

$ delta P = delta C S + delta S + S delta D. $ <eq:svd_delta_p>

Since $delta C$ and $delta D$ are skew-symmetric, they must have zero real part in diagonal elements. It immediately follows that
$
delta S = Re[I compose delta P] = I compose (U^dagger delta A V + h.c.)/2.
$ <eq:svd_delta_s>

Let us denote the complement of $I$ as $overline(I) = 1-I$. We have
$
cases(
  overline(I) compose delta C = (overline(I) compose delta P) S^(-1) - S delta D S^(-1),
  overline(I) compose delta D = S^(-1) (overline(I) compose delta P) - S^(-1) delta C S,
  I compose (delta C + delta D) = i Im[I compose delta P] S^(-1)
)
$
The last line is for determining the imaginary diagonal part of $delta C$ and $delta D$, which can not be determined from the first two lines.
Combining with @eq:svd_delta_c_d, we have

$
&cases(
  S (overline(I) compose delta P) + (overline(I) compose delta P)^dagger S &= S^2 (overline(I) compose delta D)-delta D S^2,
  (overline(I) compose delta P) S + S (overline(I) compose delta P)^dagger &= (overline(I) compose delta C) S^2-S^2 delta C
),\ 
arrow.double.r &cases(
    overline(I) compose delta D = -F compose (S delta P + delta P^dagger S),
    overline(I) compose delta C = F compose (delta P S + S delta P^dagger),
    I compose (delta C + delta D) = S^(-1) compose (delta P - delta P^dagger)/2
)
$ <eq:svd_delta_c_d_p>
where $ F_(i j) = cases(1/(s_j^2-s_i^2)\, &i != j, 0\, &i = j). $ From top to bottom, we also need to consider the contribution from the diagonal imaginary parts of $delta P$.
It is important to notice here, the imaginary diagonal parts of $delta P$ is impossible to be determined from the above equation, since they are cancelled out.
Hence, we still need the extra constraints, which is the gauge invariance of the loss function.

To wrap up, we have

$
  tr[overline(A)^dagger delta A + h.c.] &= tr[overline(U)^dagger delta U + overline(V)^dagger delta V + overline(S) delta S + h.c.]\
  &= tr[overline(U)^dagger U delta C + V S^(-1) overline(U)^dagger (I-U U^dagger) delta A + h.c.]\
  &quad - tr[overline(V)^dagger V delta D - U S^(-1) overline(V)^dagger (I-V V^dagger) delta A^dagger + h.c.]\
  &quad + tr[(overline(S) compose I) (U^dagger delta A V + h.c.)]
$ <eq:svd_loss_diff>
where we have used
$
delta U &= (U U^dagger)delta U + (I-U U^dagger)delta U = U delta C + (I-U U^dagger)delta A V S^(-1),\
delta V &= (V V^dagger)delta V + (I-V V^dagger)delta V = -V delta D + (I-V V^dagger)delta A^dagger U S^(-1).
$
The second term in the first and second line can be derived by multiplying @eq:svd_diff by $(I - U U^dagger)$ on the left and $(I - V V^dagger)$ on the right respectively.
We first consider the off-diagonal terms in @eq:svd_delta_c_d_p, and plug them into @eq:svd_loss_diff, we have
$
tr[overline(U)^dagger U (overline(I) compose delta C) + h.c.] &= tr[overline(U)^dagger U (F compose (delta P S +  S delta P^dagger)) + h.c.]\
&= tr[V S (J + J^dagger) U^dagger delta A + h.c.]
$
where $J = F compose (U^dagger overline(U))$, which has diagonal elements being all zeros.
Similarly, we have
$
-tr[overline(V)^dagger V (overline(I) compose delta D) + h.c.] &= tr[V (K + K^dagger) S U^dagger delta A + h.c.]
$
where $K = F compose (V^dagger overline(V))$.

$ tr[(S^(-1)  compose (overline(U)^dagger U - U^dagger overline(U))/2) U^dagger delta A V + h.c.] $

Now lets consider the diagonal terms in @eq:svd_delta_c_d_p, and plug them into @eq:svd_loss_diff, we have
$
&tr[overline(U)^dagger U (I compose delta C) - V^dagger V (I compose delta D) + h.c.]\
&= tr[(I compose (overline(U)^dagger U - h.c.)) delta C - (I compose (overline(V)^dagger V - h.c.)) delta D]\
$ <eq:svd_loss_diff_diag>

At a first glance, it is not sufficient to derive $delta C$ and $delta D$ from $delta P$, but consider there is still an constraint not used, *the loss must be gauge invariant*, which means

$ cal(L)(U Lambda, S, V Lambda) $

Should be independent of the choice of gauge $Lambda$, which is defined as $"diag"(e^(i phi_1), ..., e^(i phi_n))$.
Now consider a infinitesimal gauge transformation $U arrow.r U (I + i delta phi)$ and $V arrow.r V (I + i delta phi)$, where $delta phi = "diag"(delta phi_1, ..., delta phi_n)$.
When reflecting this change on the loss function, we have

$
  2 delta cal(L) = tr[overline(U)^dagger U i delta phi + overline(V)^dagger V i delta phi + "h.c."] = 0
$
which is equivalent to
$ (I compose (overline(U)^dagger U - h.c.)) + (I compose (overline(V)^dagger V - h.c.)) = 0. $

Inserting this constraint into @eq:svd_loss_diff_diag, we have
$
tr[(I compose (overline(U)^dagger U - h.c.)) (delta C + delta D)]
$
Using @eq:svd_delta_c_d_p, we have
$
&tr[(overline(U)^dagger U - h.c.)(S^(-1) compose (delta P - delta P^dagger)/2)]\
= &tr[(S^(-1) compose (overline(U)^dagger U - h.c.)/2) U^dagger delta A V + h.c.]\
$


Collecting all terms, we have
$
  tr[overline(A)^dagger delta A + h.c.] &=
  tr[V S (J + J^dagger) U^dagger delta A + h.c.]\
  &quad + tr[V S^(-1) overline(U)^dagger (I-U U^dagger) delta A + h.c.]\
  &quad + tr[V (K + K^dagger) S U^dagger delta A + h.c.]\
  &quad + tr[U S^(-1) overline(V)^dagger (I-V V^dagger) delta A^dagger + h.c.]\
  &quad + tr[(S^(-1) compose (overline(U)^dagger U - h.c.)/2) U^dagger delta A V + h.c.]\
  &quad + tr[(overline(S) compose I) (U^dagger delta A V) + h.c.]
$

Collecting all terms associated with $delta A$, we have
$
  overline(A) &= U (J + J^dagger) S V^dagger && quad triangle.small.r "from " overline(U)\
  &quad + (I-U U^dagger) overline(U) S^(-1) V && quad triangle.small.r "if" U "is not full rank"\
  &quad + U S (K + K^dagger) V^dagger && quad triangle.small.r "from " overline(V)\
  &quad + U S^(-1) overline(V)^dagger (I-V V^dagger) && quad triangle.small.r "if" V "is not full rank"\
  &quad + U (S^(-1) compose (U^dagger overline(U) - h.c.)/2) V^dagger  && quad triangle.small.r "from gauge"\
  &quad + U (overline(S) compose I) V^dagger,   && quad triangle.small.r "from " overline(S)
$
which is exactly the same as @eq:svd_loss_diff_full.



== Dominant eigenvalue@Xie2020

== Matrix inversion <matrix-inversion>

== Matrix determinant <matrix-determinant>

== LU decomposition <lu-decomposition>

== Matrix exponential <matrix-exponential>

= Differentiating ordinary differential equations <differentiating-ordinary-differential-equations>

(The adjoint state method and optimal check-pointing @Griewank1992 @Liu2021. Scalar autodiff will be mentioned.)

1. Check-pointing a long, uniform program: The optimal check-pointing method.
2. Check-pointing a short, non-uniform program: MILP method.

== Differentiating Monte Carlo simulations <differentiating-monte-carlo-simulations>

(Shixin Zhang's PhD thesis@Zhang2023

== Differentiating implicit functions <differentiating-implicit-functions>

#set text(fill: blue)
[this section is borrowed from Xingyu Zhang]
#set text(fill: black)

Considering a user-defined mapping $bold(F): RR^d times RR^n -> RR^d$ that encapsulates the optimality criteria of a given problem, an optimal solution, represented as $x(theta)$, is expected to satisfy the root condition of $bold(F)$ as follows:
$ bold(F)(x^*(theta), theta) = 0 $ <eq-Ffunction>

The function $x^*(theta): RR^n -> RR^d$ is implicitly defined. According to the implicit function theorem@Blondel2022, given a point $(x_0, theta_0)$ that satisfies $F(x_0, theta_0) = 0$ with a continuously differentiable function $bold(F)$, if the Jacobian $diff bold(F)/diff x$ evaluated at $(x_0, theta_0)$ forms a square invertible matrix, then there exists a function $x(dot)$ defined in a neighborhood of $theta_0$ such that $x^*(theta_0) = x_0$. Moreover, for all $theta$ in this neighborhood, it holds that $bold(F)(x^*(theta), theta) = 0$ and $(diff x^*)/(diff theta)$ exists. By applying the chain rule, the Jacobian $(diff x^*)/(diff theta)$ satisfies

$ (diff bold(F)(x^*, theta))/(diff x^*) (diff x^*)/(diff theta) + (diff bold(F)(x^*, theta))/(diff theta) = 0 $

Computing $diff x^* / diff theta$ entails solving the system of linear equations expressed as

$ underbrace((diff bold(F)(x^*, theta))/(diff x^*), "V" in RR^(d times d)) underbrace((diff x^*)/(diff theta), "J" in RR^(d times n)) = -underbrace((diff bold(F)(x^*, theta))/(diff theta), "P" in RR^(d times n)) $ <eq-implicit-linear-equation>

Therefore, the desired Jacobian is given by $J = V^(-1)P$. In many practical situations, explicitly constructing the Jacobian matrix is unnecessary. Instead, it suffices to perform left-multiplication or right-multiplication by $V$ and $P$. These operations are known as the vector-Jacobian product (VJP) and the Jacobian-vector product (JVP), respectively. They are valuable for determining $x(theta)$ using reverse-mode and forward-mode automatic differentiation (AD), respectively.

= Checkpointing <sec-checkpointing>
The main drawback of the reverse mode AD is the memory usage. The memory usage of the reverse mode AD is proportional to the number of intermediate variables, which scales linearly with the number of operations. The optimal checkpointing@Griewank2008 is a technique to reduce the memory usage of the reverse mode AD. It is a trade-off between the memory and the computational cost. The optimal checkpointing is a step towards solving the memory wall problem

Given the binomial function $eta(tau, delta) = ((tau + delta)!)/(tau!delta!)$, show that the following statement is true.
$ eta(tau,delta) = sum_(k=0)^delta eta(tau-1,k) $

To select a proper AD tool: source to source and operator overloading.

#figure(
  table(
    columns: (auto, auto, auto),
    [], [*Source to source*], [*Operator overloading*],
    [Primitive], [basic scalar operations], [tensor operations],
    [Application], 
    align(left)[- physics simulation], 
    align(left)[- machine learning],
    [Advantage],
    align(left)[
      - correctness
      - handles effective code
      - works on generic code
    ],
    align(left)[
      - fast tensor operations
      - extensible
    ],
    [Package],
    align(left)[
      - Tapenade@Hascoet2013
      - Enzyme@Moses2021
    ],
    align(left)[
      - Jax@Jax2018
      - PyTorch@Paszke2019
    ]
  ),
  caption: "Most of the packages listed above supports both forward and backward mode AD."
)



== Adjoint State Method

The Adjoint State Method@Plessix2006 @Chen2018 is a specific method for reverse propagation of ordinary differential equations. In research, it has been found that the reverse propagation of the derivative of the integration process is also an integration process, but in the opposite direction. Therefore, by constructing an extended function that can simultaneously trace the function value and backpropagate the derivative, the calculation of the derivative is completed in the form of inverse integration of the extended function, as shown in Algorithm 1. The description of this algorithm comes from @Chen2018, where detailed derivation can be found. Here, the symbols in the original algorithm have been replaced for better understanding. The local derivatives $(diff q)/(diff s)$, $(diff q)/(diff theta)$, and $(diff cal(L))/(diff s_n)$ in the algorithm can be manually derived or implemented using other automatic differentiation libraries. This method ensures strict gradients when the integrator is strictly reversible, but when the integration error in the reverse integration of the integrator cannot be ignored, additional processing is required to ensure that the error is within a controllable range, which will be discussed in subsequent examples.

#figure(
align(left, algorithm({
  import algorithmic: *
  Function("Adjoint-State-Method", args: ([$s_n$], [$s_0$], [$theta$], [$t_0$], [$t_n$], [$cal(L)$]), {
    Cmt[Define the augmented dynamics function]
    Function("aug_dynamics", args: ([$s$], [$a$], [$theta$]), {
      Assign([$q$], [$f(s, t, theta)$])
      Return[$q$, $-a^T (diff q)/(diff s)$, $-a^T (diff q)/(diff theta)$]
    })
    Cmt[Compute the initial state for the augmented dynamics function]
    Assign([$S_n$], [$(s_n, (diff cal(L))/(diff s_n), 0)$])
    Cmt[Perform reverse integration of the augmented dynamics]
    Assign([$(s_0, (diff cal(L))/(diff s_0), (diff cal(L))/(diff theta))$], CallI("ODESolve", (smallcaps("aug_dynamics"), [$S_n$], [$theta$], [$t_n$], [$t_0$]).join(", ")))
    Return[$(diff cal(L))/(diff s_0)$, $(diff cal(L))/(diff theta)$]
  })
})),
caption: [The continuous adjoint state method])

#figure(
  canvas({}),
  caption: [
    Using (a) checkpointing scheme and (b) reverse computing scheme to avoid caching all intermediate states. The black arrows are regular forward computing, red arrows are gradient back propagation, and blue arrows are reverse computing. The numbers above the arrows are the execution order.
    Black and white circles represent cached states and not cached states (or those states deallocated in reverse computing) respectively.
  ]
)

= Applications

Differential programming tensor networks @Liao2019 @Francuz2023

= Appendix: How to test an AD rule

For example, to test the adjoint contribution from $U$, we can construct a gauge insensitive test function:

```julia
# H is a random Hermitian Matrix
function loss(A)
    U, S, V = svd(A)
    psi = U[:,1]
    psi'*H*psi
end

function gradient(A)
    U, S, V = svd(A)
    dU = zero(U)
    dS = zero(S)
    dV = zero(V)
    dU[:,1] = U[:,1]'*H
    dA = svd_back(U, S, V, dU, dS, dV)
    dA
end
```