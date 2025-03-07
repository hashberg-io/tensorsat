# TensorSat

[![Generic badge](https://img.shields.io/badge/python-3.13+-green.svg)](https://docs.python.org/3.13/)
[![Checked with Mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](https://github.com/python/mypy)
[![PyPI version shields.io](https://img.shields.io/pypi/v/tensorsat.svg)](https://pypi.python.org/pypi/tensorsat/)
[![PyPI status](https://img.shields.io/pypi/status/tensorsat.svg)](https://pypi.python.org/pypi/tensorsat/)

A SAT/SMT Solver based on hyper-optimised tensor network contraction.

- The [tensorsat](./tensorsat) folder contains code for the future `tensorsat` Python package which was deemed to be relatively stable.
- The [notebooks](./notebooks) folder contains sub-folders with Jupyter notebooks containing feature demonstrations and experiments.
- The [notebooks/prototypes](./notebooks/prototypes) folder contains sub-folder with notebooks for feature prototypes and draft experiments.
- The [paper](./paper) folder contains the working draft of the companion paper for the library.

Contents and structure are subject to change during development.


## Install

You can install/upgrade this package from [PyPI](https://pypi.org/project/tensorsat) using pip:

```
pip install -U tensorsat
```

## Contributing

All contributions, big and small, are very appreciated!
[File an issue](./CONTRIBUTING.md/#file-an-issue) for bug reports, suggestions and questions, or [make a pull request](./CONTRIBUTING.md/#make-a-pull-request) to actively contribute to the code or documentation.

However you decide to help, please refer to the [Contributor Covenant](http://contributor-covenant.org/version/1/3/0/) Code of Conduct for what we expect from our community.

For more information, please see [CONTRIBUTING](./CONTRIBUTING.md).


## License

Multiple licensing terms apply to different parts of this repository:

- The [`tensorsat`](./tensorsat) library is licensed under [LGPL v3](./LICENSE).
- All code in the [`notebooks`](./notebooks) folder and subfolders is licensed under [GPL v3](./LICENSE).
- All rights to the companion [`paper`](./paper/) are reserved.
