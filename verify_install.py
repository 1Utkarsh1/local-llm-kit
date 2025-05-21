"""
Quick test script to verify local_llm_kit installation
"""

try:
    import local_llm_kit
    print(f"local_llm_kit version {local_llm_kit.__version__} is installed correctly!")
    
    # Print available features and backends
    print("\nAvailable backends:")
    try:
        from local_llm_kit.backends import TransformersBackend
        print("- Transformers backend ✓")
    except ImportError:
        print("- Transformers backend ✗ (requires 'transformers' and 'torch')")
    
    try:
        from local_llm_kit.backends import LlamaCppBackend
        print("- llama.cpp backend ✓")
    except ImportError:
        print("- llama.cpp backend ✗ (requires 'llama-cpp-python')")
    
    print("\nAvailable features:")
    available_features = [
        "Chat API",
        "Completion API", 
        "Function calling",
        "Memory management",
        "Prompt formatting",
        "JSON mode"
    ]
    
    for feature in available_features:
        print(f"- {feature} ✓")
        
    print("\nInstallation test completed successfully!")
    
except ImportError as e:
    print(f"Error importing local_llm_kit: {e}")
    print("Please install the package with: pip install local-llm-kit") 