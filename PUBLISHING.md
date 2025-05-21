# Publishing Guide for Local LLM Kit

This document explains how to publish new versions of Local LLM Kit to PyPI.

## Prerequisites

Before publishing a new version, you need to set up the following:

1. A PyPI account
2. GitHub repository secrets for CI/CD

## Setting up PyPI Secrets in GitHub

To enable automated publishing to PyPI via GitHub Actions, you need to add your PyPI credentials as secrets:

1. Go to your GitHub repository
2. Click on "Settings" 
3. In the left sidebar, click on "Secrets and variables" → "Actions"
4. Click the "New repository secret" button
5. Add the following secrets:
   - Name: `PYPI_USERNAME` 
   - Value: Your PyPI username
6. Add another secret:
   - Name: `PYPI_PASSWORD`
   - Value: Your PyPI token (create one at https://pypi.org/manage/account/token/)

## Version Update Process

1. Update the version number in `local_llm_kit/__init__.py`
2. Update the CHANGELOG.md file with the changes
3. Commit and push your changes
4. Create a new release on GitHub:
   - Go to the repository on GitHub
   - Click on "Releases"
   - Click "Create a new release"
   - Enter the tag version (e.g., `v0.1.1`)
   - Add release notes
   - Click "Publish release"

Once the release is published, the GitHub Actions workflow will automatically build and publish the package to PyPI.

## Manual Publishing

If you need to publish manually:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Upload to PyPI
twine upload dist/*
```

## Documentation Update

After a new release, the documentation will be automatically updated on GitHub Pages by the workflow.

## Troubleshooting

If the automated publishing fails, check:

1. The GitHub Actions logs for specific errors
2. That your PyPI token has not expired
3. That the version number in `__init__.py` is correct and has not been used before 