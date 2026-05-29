.. _contributing:

********************
Contributing to eoio
********************

We welcome contributions of all kinds — bug reports, bug fixes, new readers or
processors, and documentation improvements. Browse the
`issue tracker <https://github.com/meteor-toolkit/eoio/issues>`_ to see what is
being worked on or to open a new issue.

If you are reporting a bug, please include:

- A short, self-contained Python snippet that reproduces the problem.
- The versions of *eoio* and its key dependencies (``pip show eoio`` and ``pip list``).
- A description of what you expected to happen versus what actually happened.

Development Workflow
====================

1. Clone the repository::

    $ git clone git@github.com:meteor-toolkit/eoio.git  # SSH
    $ git clone https://github.com/meteor-toolkit/eoio.git  # HTTPS

2. Create and activate a virtual environment, then install in editable mode::

    $ python -m venv venv
    $ source venv/bin/activate  # Windows: venv\Scripts\activate
    $ pip install -e .[dev]

3. Install pre-commit hooks::

    $ pre-commit install

4. Create a branch and make your changes::

    $ git checkout -b my-feature

5. Check that the tests pass before opening a pull request::

    $ ruff check eoio
    $ pytest

If you are adding a reader or processor, see :doc:`readers/readers` and
:doc:`processors/processors` respectively.
