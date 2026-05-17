.. _contributing:

********************
Contributing to eoio
********************

.. note::

  Large parts of this document came from the `Xarray Contributing
  Guide <https://docs.xarray.dev/en/stable/contributing.html>`_.

Overview
========

We welcome your skills and enthusiasm to contribute to this project, every little bit helps! There are
numerous opportunities to contribute to *eoio* beyond just writing code.
All contributions, including bug reports, bug fixes, documentation improvements, enhancement suggestions,
and other ideas are welcome.

Where to start?
---------------

If you are brand new to *eoio* or open-source development, we recommend going
through the `GitLab "issues" tab <https://gitlab.npl.co.uk/eco/tools/eoio/-/issues>`_
to see the sort of contributions that others have made and potentially find issues that interest you!

Equally if you've just been enjoying using our tool and have happened across a bug that's slipped through
or thought of some enhancement or feature that could benefit the project, have a look at
:ref:`contributing.types` and :ref:`contributing.development_workflow` below before you get started.


.. _contributing.types:
Types of Contributions
++++++++++++++++++++++
:ref:`contributing.bug_reports`

:ref:`contributing.bug_fixes`

:ref:`contributing.implement_features`

:ref:`contributing.write_documentation`

.. _contributing.bug_reports:
Bug reports and enhancement requests
------------------------------------

Bug reports are an important part of making *eoio* more stable. Having a complete bug
report will allow others to reproduce the bug and provide insight into fixing.

Trying out the bug-producing code on the *main* branch is often a worthwhile exercise
to confirm that the bug still exists. It is also worth searching existing bug reports and
pull requests to see if the issue has already been reported and/or fixed.

Submitting a bug report
+++++++++++++++++++++++

If you find a bug in the code or documentation, do not hesitate to submit a ticket to the
`Issue Tracker <https://gitlab.npl.co.uk/eco/tools/eoio/-/issues>`_.
You are also welcome to post feature requests or merge requests.

If you are reporting a bug, please use the provided template which includes the following:

#. Include a short, self-contained Python snippet reproducing the problem.
   You can format the code nicely by using `GitHub Flavored Markdown
   <http://github.github.com/github-flavored-markdown/>`_::

      from eoio.interface import *
      ds = read(...)
      ...

#. Include the full version string of *eoio* and its dependencies. You can use the
   built in function::

      import eoio
      eoio.show_versions()
      ...

#. Explain why the current behavior is wrong/not desired and what you expect instead.

The issue will then show up to the *eoio* community and be open to comments/ideas from others.

See this `stackoverflow article for tips on writing a good bug report <https://stackoverflow.com/help/mcve>`_ .

.. _contributing.bug_fixes:
Fixing Bugs
-----------

.. _contributing.implement_features:
Implement Features
------------------

If you're looking to add a reader or processor please refer to :doc:`Readers <readers/readers>` and :doc:`Processors <processors/processors>`
respectively.

.. _contributing.write_documentation:
Write Documentation
-------------------

*eoio* could always use more documentation, whether as part of the official *eoio* docs, in docstrings,
or even on the web in blog posts, articles, and such.

.. _contributing.development_workflow:
Development Workflow
====================

Ready to contribute? Here's how to set up *eoio* for local development.

#. Make sure you have an `SSH key <https://gitlab.npl.co.uk/help/user/ssh>`_ or a `Personal Access Token <https://gitlab.npl.co.uk/help/user/profile/personal_access_tokens.md>`_ set up on your account.

#. Clone the repository locally::

    $ git clone git@gitlab.npl.co.uk:eco/tools/eoio.git  # SSH

    $ git clone https://gitlab.npl.co.uk/eco/tools/eoio.git  # PAT

#. Create a python virtual environment

Using conda::

    $ cd eoio/
    $ conda create -n your_env -k python=3.*  # version must be 3.8 or greater
    $ conda activate your_env

Using python::

    $ cd eoio/
    $ python -m venv venv
    $ . venv/bin/activate

#. Install your local copy into a virtualenv. ::

    $ pip install -e .[dev]

#. Setup pre-commit hooks::

    $ pre-commit install

#. Create a branch for local development::

    $ git checkout -b #ofissue_nameofissue

   Now you can make your changes locally.

#. When you're done making changes, check that the tests pass::

    $ ruff check eoio
    $ pytest
