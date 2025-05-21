Performance Optimization
=====================

This guide provides tips and techniques for optimizing the performance of ``local_llm_kit``.

Hardware Considerations
--------------------

GPU Selection
~~~~~~~~~~

The choice of GPU significantly impacts inference performance:

- **NVIDIA RTX 30/40 Series**: Excellent performance with consumer GPUs
- **NVIDIA A100/H100**: Enterprise-grade performance for production deployments
- **AMD GPUs**: Works with ROCm backend (limited support)

Recommended VRAM for different model sizes:

+-------------+----------------------+
| Model Size  | Recommended VRAM     |
+=============+======================+
| 7B          | 8GB+ (16GB optimal)  |
+-------------+----------------------+
| 13B         | 16GB+ (24GB optimal) |
+-------------+----------------------+
| 30B+        | 24GB+ (80GB optimal) |
+-------------+----------------------+

CPU Optimization
~~~~~~~~~~~~

For CPU inference:

- Use CPUs with AVX2/AVX512 instruction sets
- Allocate at least 16GB of RAM for medium-sized models
- Set appropriate thread counts based on CPU cores

Memory Optimization
----------------

Quantization
~~~~~~~~~

Quantizing models significantly reduces memory usage:

.. code-block:: python

   from local_llm_kit import LLMClient
   
   # Using GPTQ quantized model with Transformers backend
   client = LLMClient(
       model="llama2-7b-4bit",
       backend="transformers",
       model_path="TheBloke/Llama-2-7B-Chat-GPTQ",
       quantization_config={
           "bits": 4,
           "group_size": 128
       }
   )
   
   # Using GGUF quantized model with llama.cpp backend
   client = LLMClient(
       model="llama2-7b-q4_k_m",
       backend="llama.cpp",
       model_path="/path/to/llama-2-7b-chat.q4_k_m.gguf"
   )

Efficient KV Cache Management
~~~~~~~~~~~~~~~~~~~~~~~~~

To optimize the key-value cache:

.. code-block:: python

   client = LLMClient(
       model="llama2",
       kv_cache_config={
           "max_cache_size_mb": 1024,   # Maximum KV cache size in MB
           "enable_cache_cleaning": True  # Automatically clear old entries
       }
   )

   # For long-running applications, periodically clear the cache
   client.clear_kv_cache()

Batch Processing
-------------

Process multiple prompts efficiently with batching:

.. code-block:: python

   from local_llm_kit import LLMClient
   import concurrent.futures
   
   client = LLMClient(
       model="llama2",
       max_batch_size=32  # Set based on GPU memory
   )
   
   prompts = [
       "Write a poem about mountains.",
       "Explain quantum physics.",
       "What is the capital of France?",
       # ... more prompts
   ]
   
   # Option 1: Built-in batching
   responses = client.batch_generate(
       prompts=prompts,
       max_tokens=100
   )
   
   # Option 2: Manual parallelization with threading
   def process_prompt(prompt):
       return client.chat.completions.create(
           model="llama2",
           messages=[{"role": "user", "content": prompt}]
       )
   
   with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
       results = list(executor.map(process_prompt, prompts))

GPU Optimizations
--------------

Utilize Tensor Parallelism
~~~~~~~~~~~~~~~~~~~~~~~

For multi-GPU setups, distribute model across GPUs:

.. code-block:: python

   client = LLMClient(
       model="llama2-70b",
       tensor_parallel_size=4,  # Use 4 GPUs
       device="cuda"  # Automatically distribute across available GPUs
   )

Flash Attention
~~~~~~~~~~~

Enable flash attention for faster computation:

.. code-block:: python

   client = LLMClient(
       model="llama2",
       backend="transformers",
       use_flash_attention=True
   )

Mixed Precision
~~~~~~~~~~~~

Use FP16 or BFloat16 for faster computation:

.. code-block:: python

   client = LLMClient(
       model="llama2",
       backend="transformers",
       dtype="bfloat16"  # Or "float16" based on GPU support
   )

Streaming Optimization
------------------

For streaming responses, optimize chunk size:

.. code-block:: python

   # Balance between latency and throughput with chunk size
   for chunk in client.chat.completions.create(
       model="llama2",
       messages=[{"role": "user", "content": "Write a story"}],
       stream=True,
       chunk_token_size=16  # Smaller for lower latency, larger for better throughput
   ):
       print(chunk.choices[0].delta.content or "", end="", flush=True)

Performance Benchmarking
--------------------

Measure and optimize performance:

.. code-block:: python

   import time
   from local_llm_kit import LLMClient
   
   client = LLMClient(model="llama2")
   
   prompt = "Explain the theory of relativity in simple terms."
   
   # Warmup
   client.chat.completions.create(
       model="llama2",
       messages=[{"role": "user", "content": "Hello"}],
       max_tokens=10
   )
   
   # Benchmark
   start_time = time.time()
   response = client.chat.completions.create(
       model="llama2",
       messages=[{"role": "user", "content": prompt}],
       max_tokens=100
   )
   end_time = time.time()
   
   # Calculate metrics
   generation_time = end_time - start_time
   output_tokens = response.usage.completion_tokens
   tokens_per_second = output_tokens / generation_time
   
   print(f"Generation time: {generation_time:.2f}s")
   print(f"Output tokens: {output_tokens}")
   print(f"Tokens per second: {tokens_per_second:.2f}")

Common Performance Issues
---------------------

1. **Out of Memory**: Reduce model size, enable quantization, or increase VRAM
2. **Slow Inference**: Try mixed precision, flash attention, or a faster backend
3. **High CPU Usage**: Limit thread count or switch to GPU inference
4. **Batch Processing Bottlenecks**: Tune batch size, use async processing

Advanced Configuration
------------------

For production deployments:

.. code-block:: python

   client = LLMClient(
       model="llama2",
       
       # Memory optimization
       max_memory_mapping={
           0: "24GiB",  # GPU 0: 24GB
           1: "24GiB"   # GPU 1: 24GB
       },
       
       # Computation optimization
       compute_dtype="bfloat16",
       use_flash_attention=True,
       
       # Cache settings
       disk_cache_config={
           "enable": True,
           "cache_dir": "/path/to/cache",
           "max_size_gb": 100
       },
       
       # Thread and batch settings
       num_cpu_threads=8,
       max_batch_size=16
   ) 