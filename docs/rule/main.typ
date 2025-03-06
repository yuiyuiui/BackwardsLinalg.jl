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

== Domain of derivative <domain-of-derivative>
The result of derivative is related to the Domain of the input. That is to say, giving a complex number $z$ with $i m a g(z) = 0$, $overline(z)$ with $l o s s : RR arrow RR$ is not equal to $overline(z)$ with $l o s s : CC  arrow CC$. So even if $a$ is always a real number, mathmatically, $overline(a)$ may be a complex number with non-zero imaginary part. To avoid ambiguity, when the domain of the derivative is unclear, we denote the derivative of $l o s s(z)$  with respect to $z$ defined on $D$ as $overline(z)_D$.

It's easy to prove that 

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
Complex Version
#rulebox([

(1)
$ 
&A in CC^(m times n) , r a n k(A) =  n, b in CC^m \  
&(A,b) arrow x in CC^n = arg min ||A x-b||
$

(2)
$
  &A in CC^(m times n) , b in CC^m \  
 &(A,b) arrow a in RR = min ||A x-b||\
 & arrow a = b^(dagger) (I -U U^(dagger))b
$

Here $U = s v d(A).U$
],
[

(1)
$
&overline(b) = Q R^(- dagger) overline(x)\
&overline(A) = (b - A x)overline(x)^(dagger) R^(-1)R^(-dagger) -   Q R^(-dagger)overline(x) x^(dagger)
$
Where $A=Q R$ is the QR decomposition.

(2)
$
  & overline(b) = 2overline(a)(I - U U^(dagger))b\
  & overline(U) = -2overline(a)b b^(dagger)U\
$

Use svd_back to get $overline(A)$ from $overline(U)$
])
Proof:
(1)
$
&||A X-b||^2=(A X-b)^(dagger) (A X-b) \

&min ||A X-b||^2 arrow A^(dagger)A x=A^(dagger)b
$

And do derivative on both sides of the above formula, we get
$
  & delta A^(dagger)A X +A^(dagger) delta A X + A^(dagger)A delta x = delta A^(dagger)b+A^(dagger)delta b \
  &delta x =(A^(dagger)A)^(-1)(delta A^(dagger)b+A^(dagger)delta b-delta A^(dagger)A x-A^(dagger)delta A x)
$

And according to the complex derivative rules:
$
  &delta L=1/2 T r(overline(A)^(dagger)delta A + overline(b)^(dagger)delta b+h.c.)\
  & =1/2 T r(overline(x)^(dagger)delta x+h.c.) 
$

Then we get 
$
  &2delta L=T r(overline(x)^(dagger)(A^(dagger)A)^(-1)(delta A^(dagger)b+A^(dagger) delta b-delta A^(dagger)A x-A^(dagger)delta A x)+h.c.)\

  &=T r(overline(x)^(dagger)(A^(dagger)A)^(-1)(A^(dagger)delta b-A^(dagger)delta A x)+(b^(dagger)delta A -x^(dagger)A^(dagger)delta A)(A^(dagger)A)^(-1)overline(x)+h.c.)\

  & arrow overline(A) = -A(A^(dagger)A)^(-1)overline(x)x^(dagger) + (b-A x)overline(x)^(dagger)(A^(dagger)A)^(-1)\
  & =(b - A x)overline(x)^(dagger) R^(-1)R^(-dagger) -   Q R^(-dagger)overline(x) x^(dagger)\

  &overline(b)=overline(x)^(dagger)(A^(dagger)A)^(-1)A^(dagger)\
  &=Q R^(- dagger) overline(x)
$


(2)
$
  & A^(dagger)A x = A^(dagger)b, quad a = (A x-b)^(dagger)(A x-b)\
  & arrow S V^(dagger) x = U^(dagger)b\
  & arrow  a = b^(dagger)(b - A x) = b^(dagger)(b - U S V^(dagger)x) \
  & = b^(dagger) (I - U U^dagger) b\
$

Then 
$
  &delta a = delta b^dagger (I - U U^dagger)b +b^(dagger)(-delta U U^dagger)b + b^dagger (-U delta U^dagger)b = b^dagger (I - U U^dagger) delta b
$

Plug it and we get:
$
  & tr(overline(b)^dagger delta b + overline(U)^dagger delta U +h.c.) = 2tr(overline(a)delta a)\
  & = 2overline(a) tr(b^dagger (I-U U^dagger) delta b - U^dagger b b^dagger delta U +h.c.)\
  & arrow overline(b)^dagger = b^dagger (I-U U^dagger), quad overline(U)^dagger = - U^dagger b b^dagger\
  & overline(b) = 2overline(a)(I-U U^dagger)b, quad overline(U) = -2overline(a) b b^dagger U\
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

== Eigenvalue decomposition for hermitian and normal matrix <eigenvalue-decomposition-normal>

#rulebox([
  Eigenvalue decomposition for hermitian and normal matrix
$ A = U E U^dagger, $
  where the input $A$ is a hermitian or normal matrix, the outputs $U$ is a unitary matrix and $E$ is a diagonal matrix.


],
[
  (1) back rule for hermitian matrix eigenvalue decomposition
$
overline(A) = U[overline(E)_RR + 1/2( U^dagger overline(U) compose F + "h.c.")]U^dagger
$
where $F_(i j)=(E_j - E_i)^(-1)$.

(2) back rule for normal matrix eigenvalue decomposition
$
overline(A) = U[overline(E)_CC + 1/2( U^dagger overline(U) - overline(U)^dagger U)compose F^*]U^dagger
$
where $F_(i j)=(E_j - E_i)^(-1)$.
])

Proof: For a nomral matrix $A$, we have $A = U E U^dagger$. Then we have:
$
  &delta A = delta U E U^dagger + U delta E U^dagger + U E delta U^dagger\
  & arrow U^dagger delta A U = delta E + U^dagger delta U E + E delta U^dagger U\
  & = delta E + U^dagger delta U E - E U^dagger delta U\
  & = delta E + U^dagger delta U compose (E_j - E_i)_(n times n)\
  & arrow delta E = U^dagger delta A U compose I,quad U^dagger delta U = U^dagger delta A U compose F\
$
So:
$
  &2delta L = tr[overline(A)^dagger delta A + h.c.] = tr[overline(U)^dagger delta U + overline(E)^dagger delta E  + h.c.]\
  & = tr[overline(U)^dagger delta U + overline(U)delta U^dagger + overline(E)^dagger delta E + overline(E) delta E^dagger]\
  & = tr[overline(U)^dagger delta U + overline(U) U^dagger U delta U^dagger + overline(E)^dagger delta E + overline(E) delta E^dagger]\
  & = tr[overline(U)^dagger delta U - overline(U) U^dagger delta U U^dagger + overline(E)^dagger delta E + overline(E) delta E^dagger]\
  & = tr[U^dagger (U overline(U)^dagger - overline(U) U^dagger)U  U^dagger delta U + overline(E)^dagger delta E + overline(E) delta E^dagger]\
  & = tr[(overline(U)^dagger U - U^dagger overline(U))(U^dagger delta A U compose F) + overline(E)^dagger delta E + overline(E) delta E^dagger]\
$

If $A$ is hermitian, we have $delta E, overline(E) in RR^n$, then the above equation is equivalent to
$
  &=tr[U[-(overline(U)^dagger U - U^dagger overline(U))compose F]U^dagger delta A + 2overline(E)_RR delta E ]\
  &=tr[U[(U^dagger overline(U) - overline(U)^dagger U)compose F]U^dagger delta A + 2overline(E)_RR (U^dagger delta A U compose I)) ]\
  &=tr[U[(U^dagger overline(U) - overline(U)^dagger U)compose F + 2overline(E)_RR]U^dagger delta A]\
  &=2tr[overline(A)delta A]\
  & arrow overline(A) = U[1/2(U^dagger overline(U) - overline(U)^dagger U)compose F + overline(E)_RR]U^dagger\
$

And if $A$ is normal, we have $delta E, overline(E) in CC^n$, which means even the input $A$ is a hermitian matrix, $overline(E)_CC$ is different from $overline(E)_RR$ above.Then we have:

$
  &2delta L =tr[U[-(overline(U)^dagger U - U^dagger overline(U))compose F]U^dagger delta A + overline(E)_CC^dagger delta E + overline(E)_CC delta E^dagger]\
  &=tr[1/2U[(U^dagger overline(U) - overline(U)^dagger U)compose F]U^dagger delta A+ overline(E)_CC^dagger delta E +h.c.]\
  &=tr[1/2U[(U^dagger overline(U) - overline(U)^dagger U)compose F +overline(E)_CC^dagger]U^dagger delta A + h.c.]\
  & = tr[overline(A)^dagger delta A + h.c.]\
  & arrow overline(A) = U[1/2(U^dagger overline(U) - overline(U)^dagger U)compose F^* + overline(E)_CC]U^dagger\
$
QED.




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



== Schatten norm
#rulebox([
$ 
&A in CC^(m times n) \
&||A||_p=(sum_i lambda_i^p)^(1/p) , 1<= p< infinity\
&||A||_(infinity) = max_i lambda_i 
$
Denote $||A||_p$ as $a>= 0$.\
${lambda_i}$ are the singular values of $A$
],
[
$
& overline(A)= overline(a)a^(1-p)U S^(p-1) V^(dagger), 1<=p<infinity\
& overline(A) =overline(a)u_1 v_1^(dagger),   p=infinity

$
Where $U,S,V= s v d(A)$ and $u_1,v_1$ are respectively the first columns of $U$ and $V$.

When p=2, $||A||_2=||A||_F$ 
])

Proof: (1) 1<=p < infinity
$
A arrow S arrow a
$

Denote $S=d i a g(lambda_1,..,lambda_n)$, then
$
  &a=(sum_(i=1)^n=lambda_i^(p))^(1/p)\
  &arrow delta a = T r(a^(1-p)S^(p-1)delta S)\
  &arrow delta=T r(overline(a)delta a) = T r(overline(a)a^(1-p)S^(p-1)delta S) =T r(overline(S)delta S)\
  &arrow overline(S)=overline(a)a^(1-p)S^(p-1)\
$

And with adjoint formula of complex-value SVD, we get
$
  &overline(A)=U overline(S)V^(dagger)=overline(a)a^(1-p)U S^(p-1) V^(dagger)
$

(2) When $p=infinity$

Because singular values in $S$ got by svd() is descending order, 
$
  &a=lambda_1\
  &arrow delta a =T r (E_(11)delta S )\
  &arrow T r(overline(a)E_(11)delta S) = T r (overline(S)delta S)\
  &arrow overline(S)=overline(a)E_(11)\
  &arrow overline(A)=U overline(S) V^(dagger)=overline(a)u_1v_1^(dagger)
$

== Matrix inversion <matrix-inversion>
#rulebox([
$ 
A in CC^(n times n),det A !=0\
A->A^(-1)
$
],
[
  Denote $A^(-1)$ as $B$, then:
$
& overline(A)=-B^(dagger)overline(B)B^(dagger)
$
])

Proof: 
$
  &B A=I\
  &arrow delta B A+A delta B=0\
  &arrow delta A=-A delta B A\
  &arrow T r(-A overline(A)^(dagger)A delta B+h.c.) = T r(overline(B)^(dagger)delta B+h.c.)\
  &arrow overline(B)^(dagger)=-A overline(A)^(dagger)A \
  & arrow overline(A)=-B^(dagger)overline(B)B^(dagger)
$

== Matrix determinant <matrix-determinant>
#rulebox([
$ 
A in CC^(n times n),det A !=0\
A->a = det A
$
],
[
  Denote the adjoint matrix of $A$ as $A^(a d)$:
$
& overline(A)=overline(a)A^(a d dagger)
$
])
Proof: 
$
  &delta a=T r(A^(a d )delta A)\
  &arrow 2delta L=T r(overline(a)^* delta a +h.c.)=T r(overline(A)^(dagger)delta A+h.c.)\
  &=T r(overline(a)^* A^(a d )delta A +h.c.)\
  &arrow overline(A)=overline(a)A^(a d dagger)

$

== LU decomposition <lu-decomposition>
In some numerical package, the input matrix $A$ will be multiplied with a rows permutation matrix $P$ so that the LU decomposition of $P A$ exists. $A arrow P$ is not a map so we can't just caonsider 
$
  A arrow P L U
$

We only condider matrice that have LU decomposition. For those who can't, we have to get the $P$ and
$ A arrow P A arrow L U(P A) $

Now $A = P overline(P A)$.

#rulebox([
 
$A$ in $CC^(n times n)$ and can do LU decomposition.
$
  & A arrow L,U:L U
$
$L$ is a lower triangular matrix with all $1$ on its diagonal. $U$ is a upper triangular matrix.
],
[
$
  overline(A) = P L^(-dagger)(overline(U)U^(dagger)compose K + L^(dagger)overline(L)compose J)U^(-dagger)
$
$K$ is an upper triangular matrix with with all 1 . $J=o n e s-K$
])

Proof: First we consider $A =L U$:
$
  &A=L U\
  & arrow delta A = delta L U + L delta U\
  & arrow L^(-1)delta A U^(-1) = L^(-1) delta L +delta U U^(-1),quad delta U =L^(-1)(delta A-delta L U)
$
Because $delta U U^(-1)$ is upper triangle and $L^(-1)delta L$ lower triangle with 0 on diagonal,
$
  &L^(-1)delta L = J compose L^(-1)delta A U^(-1)\
$
Then:
$
  &T r (overline(A)^(dagger)delta A + h.c.)= T r (overline(L)^(dagger)delta L+ overline(U)^(dagger)delta U +h.c.)\
  &=T r(overline(L)^(dagger)delta L + overline(U)^(dagger)L^(-1)(delta A-delta L U)+h.c.)\
  &=T r(overline(U)^(dagger)L^(-1)delta A +(overline(L)^(dagger)L-U overline(U)^(dagger))L^(-1)delta L +h.c.)\
  &=T r(overline(U)^(dagger)L^(-1)delta A +(overline(L)^(dagger)L-U overline(U)^(dagger))(J compose L^(-1)delta A U^(-1))+h.c.)\
  & =T r(overline(U)^(dagger)L^(-1)delta A +U^(-1)  ((overline(L)^(dagger)L-U overline(U)^(dagger))compose J^T)  L^(-1)delta A+h.c.)\
  & = T r (U^(-1)  ((overline(L)^(dagger)L-U overline(U)^(dagger))compose J^T + U overline(U)^(dagger))  L^(-1)delta A+h.c.)\
  & = T r (U^(-1)  (overline(L)^(dagger)L compose J^T + U overline(U)^(dagger)compose K^T)  L^(-1)delta A+h.c.)\
  & arrow overline(A) = L^(-dagger)(overline(U)U^(dagger)compose K + L^(dagger)overline(L)compose J)U^(-dagger)
$

So for general $A$, we have :
$
  & overline(A) = P L^(-dagger)(overline(U)U^(dagger)compose K + L^(dagger)overline(L)compose J)U^(-dagger)
$

== Linear equations
#rulebox([
 $
   & A in CC^(n times n), det A !=0, b in RR^n\
   & A,b arrow x: A x =b
 $
],
[
$
& overline(A) = -A^(-dagger)overline(x)x^(dagger)\
&overline(b)=A^(-dagger)overline(x)\
$
])
Proof: 
$
  &b= A^(-1)b\
  & arrow overline(A^(-1)) = overline(x)b^(dagger) = - A^(dagger)overline(A)A^(dagger) \
  &arrow overline(A) = -A^(-dagger)overline(x)b^(dagger)A^(-dagger) = -A^(-dagger)overline(x)x^(dagger)\
  &overline(b)=A^(-dagger)overline(x)\
$


== Expmv

== Analytic matrix function <matrix-exponential>

For $A in CC^(n times n), f(z)=sum_(n=0)^(infinity) a_n z^n$ we define 
$
  &f(A)= sum_(i=1)^(infinity) a_n A^n
$

#rulebox([
$ 
A in CC^(n times n), A arrow B=f(A)
$

],
[
$
  overline(A) =sum_(n=1)^(infinity)a_n^* sum_(k=0)^(n-1)A^(dagger k)overline(B)A^(dagger (n-k-1))
$
For the unclosed form of general $A$, we turn to normal $A in C^(n times n)$,then :
$
  &overline(A)=U(overline(S)+1/2 (overline(U)^(dagger)U compose F +h.c.))U^(dagger)\

  & overline(U)=overline(B)U f(S)^(dagger)+overline(B)^(dagger)U f(S)\
  & overline(S)=f'(S)^(dagger)U^(dagger)overline(B)
$

])

Proof: 
(1) For a general $A$,
$
  & B=f(A)=sum_(n=0)^(infinity)a_n A^n\
  & delta B =sum_(n=1)a_n sum_(k=0)^(n-1)A^k delta A A^(n-1-k)
$ 

$
  & T r(overline(B)^(dagger)delta B +h.c.) = T r(overline(A)^(dagger)delta A +h.c.)\

  & = T r(overline(B)^(dagger)sum_(n=1)a_n sum_(k=0)^(n-1)A^k delta A A^(n-1-k) + h.c.)\
  & = T r(overline(B)^(dagger)sum_(n=1)a_n sum_(k=0)^(n-1)A^k overline(B)^(dagger) A^(n-1-k) delta A + h.c.)
$

$
  & arrow overline(A) =sum_(n=1)^(infinity)a_n^* sum_(k=0)^(n-1)A^(dagger k)overline(B)A^(dagger (n-k-1))
$

(2) For a normal $A$,
$
  &A arrow U,S: A = U S U^(dagger) arrow B=f(A) =U f(S) U^(dagger)\

  &delta B = delta U f(S)U^(dagger) + U f'(S) delta S U^(dagger) + U f(S) delta U^(dagger)\

  &T r(overline(U)^(dagger)delta U + overline(S)^(dagger)delta S+h.c.) = T r(overline(B)^(dagger)delta B +h.c.)\
  &= T r(overline(B)^(dagger)(delta U f(S)U^(dagger) + U f'(S) delta S U^(dagger) + U f(S) delta U^(dagger))+h.c.)\
  & T r(overline(B)^(dagger)(delta U f(S)U^(dagger) + U f'(S) delta S U^(dagger)) + delta U f(S)^(dagger)U^(dagger)overline(B) + h.c. )\

  & arrow \
  & overline(U)=overline(B)U f(S)^(dagger)+overline(B)^(dagger)U f(S)\
  & overline(S)=[f'(S)^(dagger) U^(dagger) overline(B) U] compose I
$

== Cholesky decomposition
#rulebox([
 
For a Hermite matrix $A in CC^(n times n)$, if it's positive defined, it has unique decomposition of 
$
  A = L L^(dagger)
$
where $L$ is a lower triangular matrix with real numbers on the diagonal.
],
[
  Denote $M$ as an upper triangle matrix with 0.5 on the diagonal and 1 for other nonzeros elements. Then: 
  $
   overline(A) = 1/2L^(-dagger)c o p y l t u(L^(dagger)overline(L))L^(-1)
  $
  Here, the function copyltu() means:
  $
    c o p y l t u(X) = X compose M^T +X^(dagger) compose M
  $
])
Proof: 
$
  &A=L L^(dagger)\
  &arrow delta A =delta L L^(dagger)+L delta L^(dagger)\
  &arrow L^(-1)delta A L^(-dagger) = L^(-1)delta L+delta L^(dagger)L^(-dagger)\
$
Because $L^(-1)delta L$ is an upper triangle matrix and $L^(-1)delta L+(L^(-1)delta L)^(dagger)$ is a hermite matrix, we get:
$
  &delta L^(dagger)L^(-dagger) = (L^(-1)delta A L^(-dagger))compose M\
  &delta L = (delta A-L delta L^(dagger))L^(-dagger)
$

Plug in $delta L$ we have:
$
  &2delta cal(L) = T r(overline(A)^(dagger)delta A+h.c.)=2T r(overline(A)delta A)=T r(overline(L)^(dagger)delta L+ overline(L)delta L^(dagger))\
  &=T r(L^(-dagger)overline(L)^(dagger)delta A+(L^(dagger)overline(L)-overline(L)^(dagger)L)delta L^(dagger)L^(-dagger))\
  & =T r(L^(-dagger)overline(L)^(dagger)delta A+(L^(dagger)overline(L)-overline(L)^(dagger)L) (L^(-1)delta A L^(-dagger)compose M))\
  & =T r(L^(-dagger)overline(L)^(dagger)L L^(-1)delta A+L^(-dagger)((L^(dagger)overline(L)-overline(L)^(dagger)L)compose M^T)L^(-1)delta A)\
  & =T r(L^(-dagger)(overline(L)^(dagger)L+(L^(dagger)overline(L)-overline(L)^(dagger)L)compose M^T )L^(-1)delta A)\
  & = T r(  L^(-dagger)(  overline(L)^(dagger)L compose M + L^(dagger)overline(L)compose M^T  )L^(-1)delta A  )\
  & = T r(L^(-dagger)c o p y l t u(L^(dagger)overline(L))L^(-1)delta A)\
$

$
  arrow overline(A) = 1/2L^(-dagger)c o p y l t u(L^(dagger)overline(L))L^(-1)
$



== LP

#rulebox([
Assume $P$ is a standard linear programming that has a unique optimal solution, which is a nondegenerate basic feasible solution. Then :

(Here the nondegenerate condition can be removed, but then we need more complex constraints and math proof. We now temporarily ignore this situation) 
$ 
& A in RR^(n times m), m>=n ,c in RR^m, b in RR^n\

& min c^T x\
& A x=b,x>=0

$

Denote its optimal solution is $x^0$ and the optimal value is $a$.

],
[
Denote the basic matrix related to the basic feasible solution $x$ is $B$ and it related index set in $A$ is $M = {j_1<..<j_n}$. Denote
$
  &c_B = (c_(j_k))_(k=1)^n
$
So do $overline(c)_B,x_B,overline(x)_B$. Then:

$
  &overline(B) = B^(-T)overline(x)_B x_B^T\
  &overline(b)=B^(-T)overline(x)_B\
  &overline(A)=(overline(A)_j), quad overline(A)_j = overline(B)_k (j=j_k in M) quad o r quad 0 ( o t h e r s)\
$
and

$
  &overline(x)_B = overline(a) c_B\
  &overline(c)_B = overline(a) x_B\
  &overline(c) = (overline(c)_j), quad overline(c)_j=(overline(c)_B)_k (j=j_k in M) quad o r quad 0(o t h e r s)
$
Of course, beside $M$, elements of other indices in $overline(x)_B$ are all 0.
])

Proof : $B$ is a basic matrix, so $det B !=0$. If we use $delta$ represents a slight change (CARE : not derivative operator, but an enough samll real array). Then we still have $det (B + delta B)!=0$, so the solution of
$
  (B + delta B)(x_B+ delta x_B) = b + delta b
$
is also a basic solution.

$x$ is nondegenerate $arrow x_B >0 arrow x_B+delta x_B >0$. So $x_B+delta x_B$ keeps a feasible nondegenerate solution.

Denote indices set of nonbasic variables as $N$, then $overparen(c)_N>0$. Here $overparen(c)$ is the reduced cost. Otherwise, we get $j in N$ s.t. $overparen(c)_j=0$ and we can move $x$ toward $-B^(-1)A_j$ a slight $d>0$, then $c^T x = c^T (x-d B^(-1)A_j)$, conflict with the unique optimal solution. So we still have $overparen(c)_N+delta overparen(c)_N>0$ .

Because $x_B+delta x_B$ is nondegenerate and $overparen(c)_N>0$, $x_B$ is still the unique optimal solution. 

That is to say, when change $B,b,c$ slightly, the optimal solution $x$ keeps the unique optimal solution, basic ans nondegenerate, and is only related to $B=A_M,b$.

$
  &B x_B=b arrow delta B x_B +B delta x_B =delta b arrow delta x_B=B^(-1)(delta b-delta B x_B)\
  &T r(overline(B)^T delta B+overline(b)^T delta b) = T r(overline(x)_B^T delta x_B) = T r(overline(x)_B^T B^(-1)(delta b-delta B x_B))\
  & arrow overline(B) = B^(-T)overline(x)_B x_B^T,quad overline(b)=B^(-T)overline(x)_B
$

Similarly,arroding to above adjoint formula of $C=A B$, we get
$
  & a=c_B^T x_B \
  & arrow overline(x)_B = overline(a) c_B,quad overline(c)_B = overline(a) x_B\
$
Q.E.D.
 
== GMRES

#rulebox([
Usual GMRES only works well for Diagonally Dominant Matrix. For rand(T, n, n) it can't even get a precise solution. I only give an adjoint 
for usual real and complex GMRES. It reminds to be improved.

For a large scale $A \in CC^(m times n), b in CC^m$, and fixed error $epsilon$ and initial guess $x_0$. Denote $r_0 = b - A x_0$, then we want find
$
  x in x_0 + s p a n (r_0,A r_0,..,A^(k-1)r_0) quad s.t. quad x = arg min ||b-A x||
$

We realize it by solve:
$
  y = arg l s t s q(H_k,||r_0||e_1).
$

$H_k$ comes from Schmidt Orthogonalization process:
$
  &W_k = [r_0,..,A^(k-1)r_0] arrow V_k\
  &A V_k = V_(k+1)H_k
$
Here $V_k$ is an orthonormal basis derived from $W_k$ using the Gram-Schmidt orthogonalization process.

Care that $m != n$ mean even the origin equation doesn't have a solution or its solutions are not unique, we can still get an approximate solution or one solution by GMRES.


],
[

  #strong[1. Exact AD rule:]

  Given itereation times $k$ we can do this (denote is as: GK_GMRES, G G for short) to replece usual GMRES:
  $
    &(1) A, b arrow r_0\
    &(2) A, r_0 arrow W = [r_0,..,A^k r_0]\
    &(3) W arrow Q,R = q r(W)\
    &(4) A, Q arrow H = Q'A Q[:,1:k]\
    &(4.5) H = H compose M\
    &(5) H, R arrow y = arg l s t s q (H, R[1,1]e_1)\
    &(6) x = x_0 + Q[:,1:k]y
  $

  Here $M$ is a mask matrix that:
  $
    &M = (c_(i j))_((k+1)times k), quad c_(i j) = 0 , i <=j-2\
    &c_(i j) = 1 quad f o r quad o t h e r s  
  $
  (4.5) is to make sure places in $H$ that $i<=j-2$ is $0$. Then it's adjoint:

  (1) Real:

  $
    & overline(A) = j a c(G G, A, b)[1]'overline(x)\
    & overline(b) = j a c(G G, A, b)[2]'overline(x)\
  $
  $j a c()$ means jacobian.

  (2) Complex:
  Denote:
  $
    &A = A_r + im A_i\
    &b = b_r + im b_i\
    &J A_r, J A_i, J b_r, J b_i = j a c(G G, [A_r,-A_i;A_i,A_r], [b_r;b_i])
  $
  Then:
  $
    &overline(A) = (J A_r' + im J A_i')overline(x)\
    &overline(b) = (J b_r' + im J b_i')overline(x)\
  $

  #strong[2. Approximate AD rule:]

  When $||A x - b||$ is small enough, we can approximately think $x$ is just the solution of $A x = b$ and thus we can use backrule of linear equations:
  $
    &overline(A) = -overline(b)x^(dagger)\
    &overline(b)=A^(-dagger)overline(x)\
  $

  $overline(b)$ can be got by $overline(b) = g m r e s(A',overline(x))$, which is fast.


  
])

Proof : In usual GMRES, $V_k$ is an orthonormal basis of $s p a n(W_k)$. QR decomposition do the same process.  $q r(W_k).Q$ is also an orthonormal basis of $s p a n(W_k)$. So we can replace original $H_k$ by:
$
  H_k = Q'A Q[:,1:k].
$
Then do the same derivation process of usual GMRES, we get
$
  &y = arg l s t s q (H,R[1,1]e_1).
$

== Pfaffian
#rulebox([

For $A in RR^(2n times 2n)$ and $A + A^T =0$:
$
  &P f(A)=1/(2^n n!) sum_(sigma in S_(2n)) s g n(sigma)product_(i=1)^n A_(sigma(2i-1),sigma(2i))
$

],
[
  Denote $P f(A)$ as $a$, then:
$
  &overline(A) = -(overline(a) A^(a d))/(2 a)
$
])

Proof: 
$
  &P f(A)^2 = det(A)\
  &arrow 2 P f(A) tr(((partial a)/(partial A))^T delta A ) = tr(A^(a d)delta A)\
  & arrow 2a ((partial a)/ (partial A))^T =  A^(a d)\
  & arrow overline(A) = overline(a) (partial a)/ (partial A) = -(overline(a) A^(a d))/(2 a)
$
Q.E.D.






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

