#import "macros.typ": *
#import "template.typ": template

#set text(lang: "en", region: "gb")

#show: template.with(
title: [Tensor Contraction for SMT Solving],
// abstract: [],
authors: (
(
name: "Stefano Gogioso",
organization: [Hashberg Ltd, London],
email: [
  #link("mailto:noreply@hashberg.io")[noreply\@hashberg.io]
  #counter(footnote).step() // footnote numbering '*' stars from *, but I want † for this.
  #footnote(numbering: "*")[
    To contact us about this work, please open an issue on the project's repository
    https://github.com/hashberg-io/tensorsat, using the label "question".
  ]
],
),
(
name: "Ayberk Tosun",
organization: [University of Birmingham],
email: "a.tosun@pgr.bham.ac.uk"
),
(
name: "Mirco Giacobbe",
organization: [University of Birmingham],
email:"m.giacobbe@bham.ac.uk"
),
),
accent-color: blue,
bibliography: bibliography("biblio.bib"),
)

/* Document-specific macros */

/* Product of bits */
#let Bits(n) = if n == [1] [${0,1}$] else [${0,1}^#n$]

/* Type of a relation. */
#let Type(r) = $"Type"(#r)$

/* Graph of a function. */
#let grph(f) = $"grph"(#f)$

/* Degree of a relation. */
#let deg(r) = $"deg"(#r)$

/* Shape of a relation. */
#let shp(r) = $"shp"(#r)$

/* rank of a relation. */
#let rnk(r) = $"rnk"(#r)$


#set math.equation(numbering: none)

= Introduction

In State-of-the-Art (SotA) verification of neural networks, solvers tackle verification questions on pre- and postconditions of neural networks in isolation (e.g. adversarial attacks), and offload other questions—required e.g. for the verification of neural certificates—to standard SMT solvers.
The current generation of SMT solvers is not designed to cope with such variety and complexity of theories, advanced competition benchmarks being restricted to simple forms of arithmetic (e.g. real linear arithmetic, integer linear arithmetic, bit-vectors).
For satisfiability queries that mix non-linear real-valued constraints of a physical model and digital constraints of a neural network or software model, existing solvers either discretise the problem by state-space gridding --- in the real-valued case @gao2013dreal --- or translate the problem to one about binary circuits via bitblasting ---  in the digital case @niemetz2023bitwuzla.
This brings the questions within the realm of theoretical solvability, but at the cost of a significant increase in problem size and an almost total loss of exploitable problem structure, with the practical effect of rendering the resulting problems unapproachable for all but the smallest of sizes.
To solve these issues, we propose the development of a new generation of SMT solvers, able to exploit the native high-level structures within each individual theory without the need for a lossy and expensive translation to binary arithmetic and similarly low-level languages.

Our work takes inspiration from recent breakthroughs in the efficient simulation of quantum computations: circuits which were at a point considered to be decades beyond the simulation capabilities of the current generation of supercomputers @arute2019quantum @madsen2022quantum @kim2023evidence were instead shown to be well within reach by a fundamental shift in structural perspective.
Where previous SotA simulation techniques see quantum circuits as sequences of instructions—modifying a global state as the computation progresses in a time-ordered fashion—modern tensor network techniques @gray2021hyper @liu2021closing @vincent2022jet @beguvsic2024fast endow them instead with more flexible semantics as unordered networks of interaction nodes --- complex linear tensors --- connected by information-carrying edges.
Freed from the bonds of time-ordered computation, the contraction order for tensors can be optimised, via hypergraph partitioning @schlag2022kahypar, to fit within available memory bounds, maximise parallelism and significantly reduce overall computation time --- in the case of the Google supremacy circuits @arute2019quantum, from thousands of years to less than a minute @gray2021hyper @vincent2022jet.

Hyper-optimised tensor network contraction, as it has come to be known, has been primarily developed in the context of quantum simulation, but its core principles extend far beyond that, to any context where computations can be endowed with an analogous unordered interpretation as generalised tensor networks, i.e. as diagrams in a compact closed category. Straightforward examples of such generalised tensor are those of Boolean circuits in SAT-solving and arithmetic circuits in simple forms of SMT-solving, both of which admit formalisations in the category of finite sets and relations via restrictions of well-studied diagrammatic calculi, such as the ZH calculus @backens2018zh @backens2023completeness.
Classical SAT and SMT solvers interpret formulae and circuits as computational trees, with input values flowed forward through gates to compute the output values, but this is not the only order available: circuits of functions can be more flexibly interpreted as unordered networks of relations—in this case, genuine Boolean tensors—and contracted in optimised order using a modification of established quantum techniques @peng2023arithmetic.
Furthermore, the compact closed semantics allows the drawing of a tight correspondence between satisfiability and graphical calculi for quantum computing @debeaudrap2020tensor @laakkonen2022thesis @laakkonen2023picturing @laakkonen2024graphical @carette2023compositionality, enabling the application of a range of established optimisation techniques which can be used to further simplify the computation. This is all low-hanging fruit, and the starting point for our proposal.

What we have, conceptually, is a generalised tensor network, where a heterogeneous collection of computation nodes interacts to form a complex neural certificate verification formula.
What we want is to evaluate this formula within the constraints of our available computational resources, something which is not possible at scale with current SotA methods.
What we know is that a similar problem was partially solved, for quantum computations, by switching from a functional, time-ordered perspective to a compact closed, unordered perspective on the structure of the computation.
What we need to do is to provide compact closed semantics for the various theories that come together in our formula, and apply techniques from hyper-optimised tensor network contraction to carry out our desired computation.

== Dependencies

We plan to use the following libraries as part of our solver:

- cotengra, a Python library for optimised contraction of large tensor networks @gray2021hyper, available at https://github.com/jcmgray/cotengra
- autoray, a Python library for abstraction of tensor operations, available at https://github.com/jcmgray/autoray
- KaHyPar, a Python library for optimised hypergraph partitioning @schlag2022kahypar, available at https://github.com/kahypar/kahypar
- CuPy, a Python library for NumPy-compatible GPU-accelerated computing @okuta2017cupy, available at https://github.com/cupy/cupy/
- Roaring Bitmaps, a high-performance compressed implementation of bitmaps @lemire2016consistently, many implementations available at https://roaringbitmap.org/, Python wrapper available at https://github.com/Ezibenroc/PyRoaringBitMap

== Benchmarking

We plan to benchmark our solver against the following SoTA SAT/SMT solvers:

- CaDiCal, a high-performance CDCL solver @biere2024cadical, available at https://github.com/arminbiere/cadical
- dReal, an SMT solver for non-linear theories over the reals @gao2013dreal, available at https://github.com/jmuellersift/dreal
- Bitwuzla, a SMT solver for theories involving fixed-size bitvectors, floating-point arithmetic, arrays and uninterpreted functions @niemetz2023bitwuzla, available at https://github.com/bitwuzla/bitwuzla

We plan to use benchmarks from the following conferences and competitions:

- SAT Competition 2024: https://satcompetition.github.io/2024/
- SMT-COMP 2024 https://smt-comp.github.io/2024/
- SAT 2024 https://satisfiability.org/SAT24/

= Relational Networks

In their most general form, satisfiability problems talk about the inhabitants of relations between  sets: given an implicit description of one such relation $R$, we may be tasked to determine whether $R != emptyset$, to produce some point $underline(x) in R$, to count the number of points in $R$, or even to explicitly enumerate all points $underline(x) in R$.

We consider very broad family of relational description languages, where relations are implicitly presented as the "contraction" of a (typically large) network of simpler relations, chosen from a fixed set of "generators", relations small enough to be enumerated with negligible complexity.

== Relations

When talking about a #defn[relation], we mean a finitary relation between any number of non-empty finite sets:

$
  R subset.eq
  underbrace(
    product_(j=1)^(m) X_j,
    Type(R)
  )
$

We refer to $deg(R) eqdef m$ as the #defn[degree] of the relation, to the sets $Type(R)_j eqdef X_j$ as its #defn[component types], and to the product set $Type(R)$ as its #defn[type].
We define the #defn[rank] of $R$ to be the number $rnk(r) eqdef |R|$ of points in the relation and its #defn[shape] to be the tuple $shp(R) eqdef (|X_j|)_(j=1)^(m)$ of sizes for its component types.
While it is insightful to allow arbitrary finite sets as component types, in practice we presume that each component type has been explicitly enumerated.
#footnote[Computationally, explicit enumeration of a set $X$ means fixing an inverse pair of functions, both with negligible complexity, between the set $X$ and the corresponding range ${0,...,|X|-1}$.]
#footnote[We adopted one-based indexing in this paper to improve legibility, but indexing in the implementation is zero-based. Ranges are zero-based also in this paper, in the form ${0,...,n-1}$ for $n$ elements, in accordance with common convention for modular arithmetic.]

== Functions and Values

Functions arise as a special case of relations, with additional information keeping track of which component types are input types and which are output types for the function.
We allow functions to have multiple input types and output types:

$
  f: product_(j=1)^(m) X_j arrow product_(i=1)^(n) Y_i
$

Every function $f$ has a corresponding relation $grph(f)$, known as its #defn[graph], listing its input-output pairs.
We fix a convention by which the function output types appear before the function input types:

$
  grph(f)
  eqdef
  {(y_1,...,y_n, x_1,...,x_m) | f(underline(x)) = underline(y)}
  subset.eq
  product_(i=1)^(n) Y_i times product_(j=1)^(m) X_j
$

This convention is sufficient to reconstruct a single-output function from its graph, but additional information about the number $n$ of outputs is necessary to reconstruct the function in the general case.
Values $underline(y) in product_(i=1)^(n) Y_i$ can be thought of as the special case of functions with no inputs, and they correspond to rank-1 relations, a.k.a. #defn[singletons]:

$
  {underline(y)} subset.eq product_(i=1)^(n) Y_i
$

Given a relation $R$, it is sometimes useful to consider the associated #defn[indicator function] $1_R$, mapping a point in the relation's type to a bit indicating whether the point is in the relation or not:

$
  1_R: && Type(R) & arrow && Bits(1) \
  && underline(x) & |-> && cases(
    1 "if" underline(x) in R,
    0 "otherwise"
  )
$

Because we are interested in relations, we will ultimately end up working with the graph of the indicator function, shown below:

$
  grph(1_R) = {
    (
      1_R (underline(x)),
      x_1,...,x_m
    )
  }
$

== Logical Structure

Fix component types $underline(X) = (X_1, ..., X_m)$ and consider the set of relations with those component types:

$
  "Rel"(underline(X))
  = {R subset.eq product_(j=1)^(m) X_j}
$

The set $"Rel"(underline(X))$ is a powerset, so it forms a Boolean algebra under subset inclusion $subset.eq$.
It is a complete distributive lattice under union and intersection, with the empty relation $emptyset$ and the entire set $product_(j=1)^(m) X_j$ as bottom and top elements:

$
  sect.big_(k=1)^K R_k
  & =
  {underline(x) | forall k=1,...,K "s.t." underline(x) in R_k}
  \
  union.big_(k=1)^K R_k
  & =
  {underline(x) | exists k=1,...,K "s.t." underline(x) in R_k}
$

It also has a relative complement operation:

$
  R backslash S
  = {underline(x) | underline(x) in R "and" underline(x) in.not S}
$

We will see later on that the intersection operation can be derived as a simple example of composition of relations, while the union operation corresponds to the linear structure of relational tensors.

