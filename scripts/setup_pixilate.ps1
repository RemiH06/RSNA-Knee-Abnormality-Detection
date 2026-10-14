# Prepara Pixilate: venv con torch CUDA y baja el corpus preprocesado + etiquetas LLM de Kaggle
# (scripts/download_data.py). Reanudable: si se corta, volver a correrlo salta lo que ya está.
# Uso (vía job.ps1): powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_pixilate.ps1
# 'Continue' y no 'Stop': en PowerShell 5.1 cualquier línea a stderr de un ejecutable (avisos de pip)
# se vuelve error terminante con 'Stop'. Los fallos se detectan con $LASTEXITCODE.
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot)

$token = (Get-Content .env | Where-Object { $_ -match '^KAGGLE_TOKEN=' }) -replace '^KAGGLE_TOKEN=', ''
if (-not $token) { throw 'Falta KAGGLE_TOKEN en .env' }
$env:KAGGLE_API_TOKEN = $token.Trim()

if (-not (Test-Path .venv\Scripts\python.exe)) { python -m venv .venv }
$py = Resolve-Path .venv\Scripts\python.exe

& $py -m pip install --upgrade pip
if ($LASTEXITCODE) { throw 'pip upgrade falló' }
# Sin versiones fijas: requirements.txt pide numpy 2.3 (Python >= 3.11) y Pixilate tiene 3.10.
# El wheel de torch trae su propio runtime de CUDA; no hace falta el Toolkit.
# torch y torchvision juntos desde el índice de CUDA: si timm trae torchvision de PyPI, éste
# arrastra un torch solo-CPU que reemplaza al de CUDA (pasó en la primera corrida).
& $py -c "import torch, sys; sys.exit(0 if torch.version.cuda else 1)" 2>$null
if ($LASTEXITCODE) {
    & $py -m pip uninstall -y torch torchvision 2>$null
    & $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE) { throw 'instalación de torch falló' }
}
& $py -m pip install timm pandas pyarrow scikit-learn kagglehub==1.0.2 pydicom opencv-python-headless
if ($LASTEXITCODE) { throw 'instalación de dependencias falló' }
& $py -c "import torch; assert torch.cuda.is_available(), 'sin CUDA'; print(torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE) { throw 'torch no ve la GPU' }

& $py scripts\download_data.py
if ($LASTEXITCODE) { throw 'falló la bajada de datos' }
Get-ChildItem data -Recurse -File | Where-Object { $_.Length -gt 1MB } |
    Select-Object FullName, @{n = 'GB'; e = { [math]::Round($_.Length / 1GB, 2) } } | Format-Table -AutoSize
Write-Output 'setup listo'
