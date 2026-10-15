param([switch]$ReuseSystemPackages, [switch]$WithOnnx)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if ($ReuseSystemPackages) { python -m venv --system-site-packages .venv }
    else { python -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}
$coursePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $coursePython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if ($WithOnnx) {
    & $coursePython -m pip install -r requirements-onnx.txt
    if ($LASTEXITCODE -ne 0) { throw 'ONNX dependency installation failed.' }
}
& $coursePython -m ipykernel install --prefix .venv --name object-detection --display-name 'Python (Object Detection)'
if ($LASTEXITCODE -ne 0) { throw 'Kernel registration failed.' }
Write-Host 'Ready. Run: .\.venv\Scripts\python.exe -m jupyterlab notebooks'
