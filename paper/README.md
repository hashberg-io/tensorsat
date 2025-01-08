# Companion Paper

This folder contains a working draft of the companion paper for this library, written in Typst.

## Typst

Below is a summary of how to use Typst locally.
For full documentation, see: https://github.com/typst/typst.

You can install Typst through different package managers.
Note that the versions in the package managers might lag behind the latest release.
  - Linux: View [Typst on Repology][repology]
  - macOS: `brew install typst`
  - Windows: `winget install --id Typst.Typst`

Once you have installed Typst, you can use it like this:
```sh
# Creates `file.pdf` in working directory.
typst compile file.typ

# Creates PDF file at the desired path.
typst compile path/to/source.typ path/to/output.pdf
```

You can also watch source files and automatically recompile on changes. This is
faster than compiling from scratch each time because Typst has incremental
compilation.
```sh
# Watches source files and recompiles on changes.
typst watch file.typ
```

Typst further allows you to add custom font paths for your project and list all
of the fonts it discovered:
```sh
# Adds additional directories to search for fonts.
typst compile --font-path path/to/fonts file.typ

# Lists all of the discovered fonts in the system and the given directory.
typst fonts --font-path path/to/fonts

# Or via environment variable (Linux syntax).
TYPST_FONT_PATHS=path/to/fonts typst fonts
```

For other CLI subcommands and options, see below:
```sh
# Prints available subcommands and options.
typst help

# Prints detailed usage of a subcommand.
typst help watch
```
