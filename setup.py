from setuptools import setup, find_packages
import os

# Read version from package __init__.py
with open(os.path.join('local_llm_kit', '__init__.py'), 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('"\'')
            break

# Read README for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="local-llm-kit",
    version=version,
    description="OpenAI-like interface for local LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Utkarsh Rajput",
    author_email="info@example.com",
    url="https://github.com/1Utkarsh1/local-llm-kit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "transformers": [
            "torch>=1.12.0",
            "transformers>=4.25.0",
            "accelerate>=0.16.0",
        ],
        "llamacpp": [
            "llama-cpp-python>=0.1.77",
        ],
        "all": [
            "torch>=1.12.0",
            "transformers>=4.25.0",
            "accelerate>=0.16.0",
            "llama-cpp-python>=0.1.77",
        ],
    },
    entry_points={
        "console_scripts": [
            "local-llm-kit=local_llm_kit.cli:main",
        ],
    },
) 