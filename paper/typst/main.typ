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

/* Graph of a function. */
#let grph(f) = $"grph"(#f)$

/* Relations with given shape. */
#let Rel(X) = $"Rel"(#X)$


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

In their most general form, satisfiability problems talk about the inhabitants of relations between  sets: given an implicit description of one such relation $R$, we may be tasked to determine whether $R != emptyset$, to produce some point $underline(x) in R$, to count the number of tuples in $R$, or even to explicitly enumerate all tuples $underline(x) in R$.

We consider very broad family of relational description languages, where relations are implicitly presented as the "contraction" of a (typically large) network of simpler relations, chosen from a fixed set of "generators", relations small enough to be enumerated with negligible complexity.

== Relations

We define a #defn[shape] to be a finite family $underline(X) = (X_j)_(j in J)$ of non-empty finite sets.
We refer to the indices $j in J$ as the #defn[components], to the sets $X_j$ as the #defn[component sets] and to $J$ as the #defn[index set].
We define the following shorthand for the product set associated to the shape:

$
  product underline(X) eqdef product_(j in J) X_j
$

We define a #defn[relation of shape $underline(X)$] to be a finitary relation with the component sets as its domains:
$
  R subset.eq product underline(X)
$

By extension, we refer to the indices $j in J$ as the #defn[components] of the relation, to the sets $X_j$ as its #defn[component sets] and to $J$ as its #defn[index set].
We define the #defn[rank] of the relation to be the number $|R|$ of tuples in it.


== Functions and Values

Functions arise as a special case of relations, with additional information keeping track of which components for the relation are inputs for the function and which ones are outputs.
We allow functions to have multiple inputs and outputs, i.e. we explicitly factorise its domain and codomain into products:

$
  f: product_(j in J) X_j times product_(o in O) Y_o
$

Every function $f$ has a corresponding relation $grph(f)$, known as its #defn[graph], listing its input-output pairs:
$
  grph(f)
  eqdef
  {underline(y) union.sq underline(x) | f(underline(x)) = underline(y)}
  subset.eq
  product underline(X) times product underline(Y)
$

where $underline(y) union.sq underline(x)$ is the following family, indexed by the disjoint product $J union.sq O$:

$
  (underline(y) union.sq underline(x))_(0, j) &= x_j \
  (underline(y) union.sq underline(x))_(1, o) &= y_o
$

Values $underline(y) in product underline(Y)$ can be thought of as the special case of functions with no inputs, and they correspond to rank-1 relations, a.k.a. #defn[singletons]:

$
  {underline(y)} subset.eq product underline(Y)
$

Given a relation $R$ of shape $underline(X)$, it is sometimes useful to consider the associated #defn[indicator function] $1_R$, mapping a tuple in the product of the relation's component sets to a bit indicating whether the tuple is in the relation or not:

$
  1_R: && product underline(X) & arrow.long && {0, 1} \
  && underline(x) & |-> && cases(
    1 "if" underline(x) in R,
    0 "otherwise"
  )
$

Because we are interested in relations, we will ultimately end up working with the graph of the indicator function.

== Boolean Structure

Fix a shape $underline(X)$ and consider the set of relations of that shape:

$
  Rel(underline(X))
  = {R subset.eq product underline(X)}
$

The set $Rel(underline(X))$ is a powerset, so it forms a Boolean algebra under subset inclusion $subset.eq$.
It is a complete distributive lattice under union and intersection, with the empty relation $emptyset$ and the entire product set $product underline(X)$ as bottom and top elements, respectively:

$
  sect.big_(k in K) R_k
  & =
  {underline(x) | forall k in K "s.t." underline(x) in R_k}
  \
  union.big_(k in K) R_k
  & =
  {underline(x) | exists k in K "s.t." underline(x) in R_k}
$

It also has a relative complement operation:

$
  R backslash S
  = {underline(x) | underline(x) in R "and" underline(x) in.not S}
$

We will see later on that the intersection operation can be derived as a simple example of contraction of relational networks.

== Wiring diagrams

We define a #defn[wiring diagram] $Delta = (K, underline(I), O, W, w^"in", w^"out")$ to consist of the following data:

- A finite set $K$ of #defn[input slots].
- A family $underline(I)$ of sets $I_k$ of #defn[input components] for each input slot $k in K$.
- A finite set $O$ of #defn[output components].
- A finite set $W$ of #defn[wiring nodes].
- An #defn[input wiring function] $w^"in"$, mapping each input component to a wiring node:

$
  w^"in": product.co_(k in K) I_k arrow W
$

- An #defn[output wiring function] $w^"out"$, mapping each output component to a wiring node:

$
  w^"out": O arrow W
$

The data is subject to the requirement that $w^"in"$ and $w^"out"$ be jointly surjective, i.e. that $im(w^"in") union im(w^"out") = W$.
We define a #defn[typed wiring diagram] $(Delta, underline(underline(X)), underline(Y))$ to consist of the following data:

- A wiring diagram $Delta = (K, underline(I), O, W, w^"in", w^"out")$.
- A family $underline(underline(X))$, indexed by the input slots $k in K$, of #defn[input shapes] $underline(X)_k$. We refer to the component sets in the input shapes as #defn[input component sets].
- An #defn[output shape] $underline(Y)$. We refer to the component sets in the output shape as #defn[output component sets].

The data is subject to the following requirements:

- Each input shape $underline(X)_k$ is indexed by the corresponding set $I_k$ of input components.
- The output shape $underline(Y)$ is indexed by the set $O$ of output components.
- All input and output components wired onto the same wiring node have the same component set. That is, we can associate a (necessarily unique) non-empty finite set $Z_nu$ to each wiring node $nu in W$ such that $X_(k,i) = Z_nu$ for all input wires $(k, i) in (w^"in")^(-1)(nu)$ and $Y_o = Z_nu$ for all output wires $o in (w^"out")^(-1)(nu)$. We refer to $underline(Z)$ as the family of #defn[wiring sets] for the wiring diagrams.

Wiring diagrams have the structure of a symmetric operad:

- Permuting the input slots of a wiring diagram results in another wiring diagram, giving rise to a right action of the symmetric group $S_K$ onto the set of wiring diagrams with input slots $K$.
- Wiring diagrams can be composed by "gluing" a suitable family of wiring diagrams $(Delta^((k)))_(k in K)$ into the input slots of a wiring diagram $Delta$ with input slots $K$, resulting in a wiring diagram with input slots $product.co_(k in K) K^((k))$, where we denoted by $K^((k))$ the input slot set for wiring diagram $Delta^((k))$.
- The composition operation is equivariant between the right action of the symmetric group $S_K$ on the input slots of $Delta$ and the right action of the direct product group $product_(k in K) S_(K^((k)))$ onto the input slots of the wiring diagrams $(Delta^((k)))_(k in K)$.

The symmetric operad structure extends to typed wiring diagrams, by enforcing compatibility of input and output shapes during composition.
For more information, please refer to @spivak2013operad and Chapter 7 of @yau2018operads (bearing in mind differences in nomenclature).


== Relational Networks

We define a #defn[relational network] $Gamma = (Delta, underline(underline(X)), underline(Y), underline(R))$ to consist of the following data:

- A typed wiring diagram $(Delta, underline(X), underline(R))$, where we write $Delta = (K, underline(I), O, W, w^"in", w^"out")$.
- A family $underline(R)$ of #defn[network relations], indexed by the input slots $k in K$ of the wiring diagram, each relation $R_k$ having input shape $underline(X)_k$.

In analogy with relations, we refer to the indices $o in O$ as the #defn[output components] for the relational network $Gamma$, to the sets $Y_o$ as its #defn[output component sets], and to the set $O$ as its #defn[output index set].
We don't define the rank of a relational network, because this information is not typically available without contracting the network into a relation.

We define the #defn[contraction] $floor(Gamma)$ of one such relational network $Gamma$ to be the relation of output shape $underline(Y)$ obtained by enforcing the existence of a compatible values for all network relations and all wiring nodes:

#[
  #show math.equation: set text(size: 10pt)
  $
    floor(Gamma) eqdef {
      underline(y) in product underline(Y)
      mid(|)
      exists underline(underline(x)) in product_(k in K) R_k. #h(3pt)
      exists underline(z) in product_(nu in W) Z_nu. #h(3pt)
      forall k in K. #h(3pt)
      forall i in I_k. #h(3pt)
      x_(k,i) = z_(w^"in" (k, i))
      text("and")
      forall o in O. #h(3pt)
      y_o = z_(w^"out" (o))
    }
  $
]

where $underline(Z)$ are the wiring sets for the typed wiring diagram.

// Give examples of common operations which can be expressed as relational network contractions:
// - Function composition (single-input, single-output)
// - Function composition (multiple-input, single-output)
// - Function composition (multiple-input, multiple-output)
// - Functions in parallel
// - Simple equations
// - Relational intersections
// - Systems of equations
// - Finding all satisfying assignments of a CNF formula
// - Determining whether a CNF formula has a satisfying assignment
