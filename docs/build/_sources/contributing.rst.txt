Contributing
===========

We love your input! We want to make contributing to ``local_llm_kit`` as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

Development Process
----------------

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

Pull Request Process
----------------

1. Fork the repository and create your branch from ``main``.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Issue that pull request!

Getting Started
------------

To set up local development:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/1Utkarsh1/local-llm-kit.git
   cd local-llm-kit
   
   # Set up a virtual environment (optional but recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install the package in development mode
   pip install -e .[all]
   
   # Install development dependencies
   pip install pytest black

Running Tests
----------

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run specific tests
   pytest tests/test_function_calling.py

Code Style
--------

We use ``black`` for code formatting. Please ensure your code follows this style:

.. code-block:: bash

   # Format code
   black .

Submitting Changes
---------------

Please send a GitHub Pull Request with a clear list of what you've done. We can always use more test coverage. Please make sure all of your commits are atomic (one feature per commit).

Always write a clear log message for your commits. One-line messages are fine for small changes, but bigger changes should look like this:

.. code-block:: bash

   $ git commit -m "A brief summary of the commit
   > 
   > A paragraph describing what changed and its impact."

License
------

By contributing, you agree that your contributions will be licensed under the project's MIT License.

Code of Conduct
------------

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms.

Contact
------

If you have any questions or need assistance, please open an issue or reach out to the maintainer at utkarshrajput815@gmail.com. 