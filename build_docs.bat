@echo off
echo Building documentation...

if not exist "docs\build" mkdir docs\build
if not exist "docs\build\html" mkdir docs\build\html

cd docs
echo "Checking for Sphinx installation..."
pip show sphinx >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo "Sphinx not found, installing..."
    pip install sphinx sphinx_rtd_theme
)

cmd /c make.bat html

if %ERRORLEVEL% neq 0 (
    echo Error building documentation!
    exit /b %ERRORLEVEL%
)

echo Documentation built successfully!
echo You can view it at docs/build/html/index.html 