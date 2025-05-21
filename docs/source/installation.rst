Installation
============

This guide will help you install ``local_llm_kit`` and set up your environment.

Requirements
----------

- Python 3.8 or higher
- pip (Python package installer)

Basic Installation
---------------

You can install the package directly from PyPI:

.. code-block:: bash

   pip install local-llm-kit

This will install the core package with minimal dependencies.

Installing Optional Dependencies
----------------------------

For using specific backends, you can install the package with extra dependencies:

.. code-block:: bash

   # For Transformers backend
   pip install "local-llm-kit[transformers]"
   
   # For llama.cpp backend
   pip install "local-llm-kit[llamacpp]"
   
   # For all features
   pip install "local-llm-kit[all]"

Development Installation
--------------------

For development purposes, you can install the package in editable mode:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/1Utkarsh1/local-llm-kit.git
   cd local-llm-kit
   
   # Install in development mode
   pip install -e ".[all]"

GPU Support
---------

For GPU acceleration with the Transformers backend:

1. Make sure you have a CUDA-compatible GPU
2. Install the appropriate CUDA toolkit for your system
3. Install PyTorch with CUDA support:

   .. code-block:: bash

      # Example for CUDA 11.8
      pip install torch --index-url https://download.pytorch.org/whl/cu118

Verification
----------

You can verify your installation with:

.. code-block:: python

   from local_llm_kit import LLMClient
   
   # This should work if installation was successful
   client = LLMClient(model="llama2")
   print(f"Successfully initialized client for model: {client.model}")

Troubleshooting
------------

Common installation issues:

1. **Missing CUDA**: If you're getting CPU-only execution despite having a GPU, check that PyTorch was installed with CUDA support:

   .. code-block:: python
   
      import torch
      print(f"CUDA available: {torch.cuda.is_available()}")
      print(f"CUDA devices: {torch.cuda.device_count()}")

2. **ImportError**: If you get an import error for one of the backends, make sure you installed the corresponding extra dependencies.

3. **Version Conflicts**: If you encounter version conflicts, try creating a fresh virtual environment:

   .. code-block:: bash
   
      python -m venv llm_env
      source llm_env/bin/activate  # On Windows: llm_env\Scripts\activate
      pip install "local-llm-kit[all]" 