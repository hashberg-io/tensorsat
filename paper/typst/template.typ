/*
 * template.typ
 * revised 2024-12-18
 */

/* Alias for the builtin 'bibliography function' */
#let builtin-bibliography = bibliography


/* A simple paper template. */
#let template(
  title: [Paper Title], // paper title
  authors: (), // array of authors: {name, organisation?, location?, email?}
  abstract: none, // paper abstract (optional)
  bibliography: none, // result of calling builtin 'bibliography' (optional),
  appendix: none, // paper appendix
  accent-color: none,
  body // paper body
) = {
  let link-color = if (accent-color == none) {black} else {accent-color}

  // Set document properties:
  set document(title: title, author: authors.map(author => author.name)) // meta
  set page(paper: "a4", margin: (x: 1in, y: 1in)) // page size and margins
  set page(numbering: "1") // page numbering
  set par(justify: true, first-line-indent: 1em) // paragraph alignment & indent
  set par(spacing: 1em)// paragraph spacing
  set text(font: "New Computer Modern", size: 12pt) // body font
  show link: set text(fill:link-color) // url links
  show ref: set text(fill:link-color) // reference links
  show cite: set text(fill:link-color) // citation links
  set heading(numbering: "1.1.1.") // section headings
  set enum(numbering: "1.a.i.") // enumerate numbering
  set enum(indent: 10pt, body-indent: 6pt) // enumerate indent
  set list(indent: 10pt, body-indent: 6pt) // itemize indent
  show raw: set text(font: "Fira Code", size: 10pt, weight:400) // code font
  show raw: set block(spacing: 1.5em) // code block spacing
  set math.equation(numbering: "(1)") // equation numbering
  show math.equation: set block(spacing: 1.5em) // equation spacing
  set figure(placement: none) // figures placement (remain in-flow)
  show figure.caption: set text(size: 12pt) // figure caption font
  set builtin-bibliography(style: "ieee") // bibliography style
  show builtin-bibliography: set text(10pt) // bibliography font

  // Display the paper title:
  align(center, text(20pt, title))

  // Display the authors list:
  v(10mm, weak: true) // spacing before authors list
  align(center)[
  #for i in range(calc.ceil(authors.len() / 3)) {
    let end = calc.min((i + 1) * 3, authors.len())
    let is-last = authors.len() == end
    let slice = authors.slice(i * 3, end)
    grid(
      // stroke: black,
      columns: slice.len() * (1fr,),
      gutter: 8mm,
      ..slice.map(author => align(center, {
        set text(12pt)
        [#author.name]
        if "organization" in author [ \ #author.organization ]
        if "location" in author [ \ #author.location ]
        if "email" in author {
          if type(author.email) == str [
            \ #link("mailto:" + author.email)
          ] else [
            \ #author.email
          ]
        }
      }))
    )
    if not is-last {
      v(8mm, weak: true)
    }
  }
  ]
  v(10mm, weak: true) // spacing after authors

  // Display abstract:
  if abstract != none [
    #set text(11pt, weight: 400)
    ABSTRACT. #abstract
  ]

  // Display the body:
  body

  // Display the bibliography:
  bibliography

  // Optionally display the appendix:
  if appendix != none {
    counter(heading).update(0)
    set heading(numbering: "A.1.1.", supplement:[Appendix])
    appendix
  }
}
