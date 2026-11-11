# Offline tasks (no paid API call). Usage from the repo root:
#   .\tasks.ps1 test         # run the unit tests
#   .\tasks.ps1 reproduce    # rebuild every result from the cached files
#   .\tasks.ps1 all          # both
param([ValidateSet("test", "reproduce", "all")][string]$Task = "all")
$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Push-Location $PSScriptRoot
try {
    if ($Task -in "test", "all") { & $py -m pytest -q; if ($LASTEXITCODE) { exit $LASTEXITCODE } }
    if ($Task -in "reproduce", "all") { & $py -m zerolabel.reproduce --seeds 0 1 2 3; if ($LASTEXITCODE) { exit $LASTEXITCODE } }
} finally { Pop-Location }
