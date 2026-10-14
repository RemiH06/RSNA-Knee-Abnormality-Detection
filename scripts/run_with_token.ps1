# Corre un script de Python del venv con KAGGLE_API_TOKEN tomado de .env (para job.ps1 en Pixilate).
# Uso: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_with_token.ps1 scripts\download_archive.py
param([Parameter(Mandatory)][string]$Script)
Set-Location (Split-Path $PSScriptRoot)
$token = (Get-Content .env | Where-Object { $_ -match '^KAGGLE_TOKEN=' }) -replace '^KAGGLE_TOKEN=', ''
if (-not $token) { throw 'Falta KAGGLE_TOKEN en .env' }
$env:KAGGLE_API_TOKEN = $token.Trim()
& .venv\Scripts\python.exe $Script
exit $LASTEXITCODE
