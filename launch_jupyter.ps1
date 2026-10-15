$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$coursePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $coursePython)) { throw 'Run setup_windows.ps1 first.' }
& $coursePython -m jupyterlab notebooks
