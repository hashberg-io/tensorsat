/*
 * macros.typ
 * revised 2024-12-18
 */

// == Math macros ==

/* If and only if */
#let iff = $arrow.double.l.r$

/* Defined as equal to */
#let eqdef = $limits(=)^"def"$

/* Defined as equivalent to */
#let equivdef = $limits(equiv)^"def"$

/* Defined as if and only if */
#let iffdef = $limits(#iff)^"def"$

/* Domain of a function */
#let dom(f) = $"dom"(#f)$

/* Codomain of a function */
#let cod(f) = $"cod"(#f)$

/* Restriction of a function to a set */
#let restrict(f, X) = $#f|_#X$

/* Set of the natural numbers */
#let naturals = $bb(N)$

/* Set of the integer numbers. */
#let integers = $bb(Z)$

/* Set of the real numbers. */
#let reals = $bb(R)$

/* Set of the complex numbers */
#let complexes = $bb(C)$

/* Slashed vertical bar */
#let slashbar = stack("/", "|", spacing: -1.7mm, dir: ltr)

// == Text macros ==

/* Centered block note with left aligned text. */
#let blocknote(
  fill: luma(0xee),
  inset: 6pt,
  radius: 4pt,
  width: 75%,
  body
) = [
  #set align(center)
  #block(
    fill: fill,
    inset: inset,
    radius: radius,
    width: width,
  )[
    #set align(left)
    #body
  ]
]

/* Marks definitions of concepts. */
#let defn(concept) = [*#concept*]

/* Marks content as work in progress. */
#let wip(body) = {
  highlight(fill:luma(0xee))[#emoji.construction #body #emoji.construction]
}

/* Marks content as describing something to be done. Visually same as wip. */
#let todo(body) = {
  // wip[*TODO:* #body]
  wip[#body]
}
