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


= Notations
Something should be careful:

1. For $z = x + i y$, 
$
  overline(x) != overline(z)|_(y=0)
$

But for Lp norm loss function these two don't make difference.

2. For a symmetric matrix input $A$, "$A$ is an input matrix" is not equal to "$A$ is a symmetric input matrix". To do the latter we shoule replace $overline(A)$ with $(overline(A) + overline(A)^(dagger))/2$

= Matrix multiplication <matrix-multiplication>
DONE

= Tensor network contraction <tensor-network-contraction>
DONE

= The least square problem <least-square-problem>
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



= QR decomposition <qr-decomposition>
1. about with pivoting: this problem is similar to LU decomposition. The process is not a map, so we can't just express $overline(A)$ with $overline(P),overline(Q),overline(R)$. We have to get the $P$ artificially and:
$
  &A arrow A P arrow  q r(A P)
$

2. For $A in CC^(m times n)$ and $r a n k(A)=n$ , the formula and calculation process keep the same because they don't use the form $Q^(-1) $ or $overline(Q)^(-1)$.

3. For $A in CC^(m times n), m<=n$, then we can get $R^r in R^(n times m)$ s.t. $R R^r = I_m$. $R^r$ can be get easily by applying the same column translation on both $R$ and $I_n$ until $A$ turns into $(I_m,0)$. $R^r$ satisfies that: denote the place of the first nonzero element on the $i_(t h)$ row of $R$ is $1<=i_1<..<i_m$, then the $(i,j)$ element of $R^r$ can be nonzero only when :
$
  &i=i_k in {i_1,..,i_m}, j>=k
$

Besides, it's easy to prove such $R^r$ in unique.

= Eigenvalue decomposition <eigenvalue-decomposition>
This adjoint formula of hermite imput is just the adjoint formula for normal matrices input.


= Singular value decomposition <singular-value-decomposition>

DONE

= Schatten norm
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

= Matrix inversion <matrix-inversion>
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

= Matrix determinant <matrix-determinant>
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

= LU decomposition <lu-decomposition>
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

= Linear equations
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


= Expmv

= Analytic matrix function <matrix-exponential>

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

= Cholesky decomposition
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



= LP

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
 

= SDP

#rulebox([
In SDP, problem on real is much different from complex one. So we discuss them respectively.

Here after, we denote the index set of basic cone as $M$ and realated $(b_i)_(i in M)$ as $b_B$. And denote $v(X)=[X[1:n,1];X[2:n,2];..;X[n,n]]$. $J$ is an upper triangle matrix with all nonzero elements being 1, and $K=(1)_(n times n)-J$. Then we solve such 2 problems:

(1)
$
  &{A_i} in RR^(n times m)  (m>=n), b in RR^(n), C in RR^(n times n)\
  & min T r(C X)\
  & T r(A_i X) = b_i\
  & X>=0
$
Assume this problem has unique nondegenerate positive defined solution and its critical cone has positive measure in its tangent space.


],
[(1)

  Do Cholesky decomposition on $X=L L^T$. Denote :
  $
    D = (v^T (L A_i))_(i in M)
  $
 Then
  $
    & overline(b)_B = overline(D)^(-T)v((overline(X)L)compose J^T )\
    & overline(A_i) = -overline(b)[i]X, quad i in M
  $
])

Proof: 
$
  &A arrow L:A=L L^(dagger) arrow X arrow arrow a= T r(C X)
$

$
  &forall i in M, T r (A_i X)=b_i \
  &arrow T r(X delta A_i+A_i delta L L^T + A_i L delta L^T )= T r(X delta A_i + 2L^T A_i delta L) =delta b_i\
  &arrow  2 v^T (L A_i)v(delta L) = delta b_i - T r(X delta A_i)\
  &arrow 2(v^T (L A_i))_(i in M) delta v(L) = delta b_B - (T r(X delta A_i))_(i in M)\
  & delta v(L) = 1/2 D^(-1)(delta b_B-(T r(x delta A_i))_(i in M))\
$

$
  & arrow T r(overline(L)^T delta L) = T r(sum_(i in M)overline(A_i)^T delta A_i + overline(b)_B^T delta d_B) = v^T (overline(L))delta v(L) \
  &= 1/2 v^T (overline(L))D^(-1)(delta b_B - (T r(x delta A_i))_(i in M))\
$

$
  & arrow overline(b)_B =1/2 D^(-T) v(overline(L)) = D^(-T) v((overline(X)L)compose J^T)\
  & overline(A_i) = -overline(b)[i]X, quad i in M

$


= GMRES

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

])

Proof : In usual GMRES, $V_k$ is an orthonormal basis of $s p a n(W_k)$. QR decomposition do the same process.  $q r(W_k).Q$ is also an orthonormal basis of $s p a n(W_k)$. So we can replace original $H_k$ by:
$
  H_k = Q'A Q[:,1:k].
$
Then do the same derivation process of usual GMRES, we get
$
  &y = arg l s t s q (H,R[1,1]e_1).
$