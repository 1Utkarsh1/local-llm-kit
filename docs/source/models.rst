Supported Models
===============

This page documents the models supported by ``local_llm_kit`` and their configurations.

Model Backends
------------

Transformers Backend
~~~~~~~~~~~~~~~~~~

The Transformers backend supports models from Hugging Face's Transformers library.

Supported Model Types:
    - LLaMA and LLaMA-2
    - Mistral
    - Falcon
    - MPT
    - GPTQ quantized models

Configuration:

.. code-block:: python

   client = LLMClient(
       model="llama2",
       backend="transformers",
       model_path="meta-llama/Llama-2-7b-chat-hf",
       device="cuda",  # or "cpu"
       dtype="float16",  # or "float32", "bfloat16"
       trust_remote_code=True
   )

llama.cpp Backend
~~~~~~~~~~~~~~~

The llama.cpp backend supports GGUF format models.

Supported Features:
    - 4-bit, 5-bit, and 8-bit quantization
    - GPU acceleration
    - Metal support on macOS
    - Efficient CPU inference

Configuration:

.. code-block:: python

   client = LLMClient(
       model="llama2",
       backend="llama.cpp",
       model_path="/path/to/model.gguf",
       n_gpu_layers=32,  # Number of layers to offload to GPU
       n_ctx=2048,  # Context window size
       n_batch=512  # Batch size for prompt processing
   )

Model Configuration
-----------------

Common Parameters
~~~~~~~~~~~~~~~

These parameters work with all model backends:

.. code-block:: python

   client = LLMClient(
       model="llama2",
       temperature=0.7,  # Randomness in generation (0.0 to 1.0)
       top_p=0.9,  # Nucleus sampling parameter
       top_k=40,  # Top-k sampling parameter
       repetition_penalty=1.1,  # Penalty for repeating tokens
       max_tokens=100,  # Maximum tokens to generate
   )

Memory Requirements
~~~~~~~~~~~~~~~~

Approximate memory requirements for different model sizes:

+-------------+------------------+------------------+
| Model Size  | FP16 (GPU)      | 4-bit Quantized |
+=============+==================+==================+
| 7B          | ~14 GB          | ~4 GB           |
+-------------+------------------+------------------+
| 13B         | ~26 GB          | ~7 GB           |
+-------------+------------------+------------------+
| 70B         | ~140 GB         | ~35 GB          |
+-------------+------------------+------------------+

Performance Tips
--------------

GPU Acceleration
~~~~~~~~~~~~~~

For optimal GPU performance:

1. Use CUDA devices when available
2. Enable flash attention if supported
3. Use appropriate batch sizes
4. Monitor GPU memory usage

.. code-block:: python

   client = LLMClient(
       model="llama2",
       device="cuda",
       use_flash_attention=True,
       max_batch_size=32
   )

CPU Optimization
~~~~~~~~~~~~~

For CPU inference:

1. Use quantized models
2. Set appropriate thread count
3. Enable CPU optimizations

.. code-block:: python

   client = LLMClient(
       model="llama2",
       device="cpu",
       threads=8,
       use_mmap=True,
       use_avx2=True
   )

Model Selection Guide
------------------

Choosing the right model depends on your use case:

1. Resource-Constrained Environments
   - Use 4-bit quantized 7B models
   - Consider CPU-optimized models
   - Reduce context length if possible

2. High-Performance Requirements
   - Use larger models (13B+)
   - Enable GPU acceleration
   - Optimize batch processing

3. Balanced Setup
   - Use 7B models with 8-bit quantization
   - Balance GPU/CPU usage
   - Adjust parameters based on workload

Custom Model Integration
---------------------

You can integrate custom models by:

1. Converting to GGUF format for llama.cpp
2. Using Hugging Face's model format
3. Implementing custom tokenizers

Example:

.. code-block:: python

   from local_llm_kit import LLMClient, CustomTokenizer

   # Custom tokenizer implementation
   class MyTokenizer(CustomTokenizer):
       def encode(self, text):
           # Implementation
           pass
           
       def decode(self, tokens):
           # Implementation
           pass

   # Use custom model
   client = LLMClient(
       model="custom",
       tokenizer=MyTokenizer(),
       model_path="/path/to/custom/model"
   )

Troubleshooting
-------------

Common Issues:

1. Out of Memory
   - Reduce batch size
   - Use quantization
   - Decrease context length

2. Slow Performance
   - Check device utilization
   - Optimize model parameters
   - Consider model quantization

3. Model Loading Errors
   - Verify model path
   - Check format compatibility
   - Ensure sufficient resources 