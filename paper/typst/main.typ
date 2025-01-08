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

- SAT Competition 2023: https://satcompetition.github.io/2023/
- SMT-COMP 2023 https://smt-comp.github.io/2023/
- SAT 2023 https://satisfiability.org/SAT23/
