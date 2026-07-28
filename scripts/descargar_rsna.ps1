# Descarga RSNA Pneumonia Detection Challenge y construye el manifiesto.
#
# Requisitos previos (los tienes que hacer tú, no se pueden automatizar):
#   1. Cuenta de Kaggle y token de API: kaggle.com/settings -> API -> Create New Token
#      (descarga kaggle.json; déjalo en Descargas o en %USERPROFILE%\.kaggle\)
#   2. Aceptar las reglas de la competición, con sesión iniciada:
#      https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/rules
#
# Uso:  .\scripts\descargar_rsna.ps1

$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent $PSScriptRoot
Set-Location $raiz

$py = Join-Path $raiz ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "No encuentro el entorno virtual en .venv. Crea con: python -m venv .venv" }

# --- 1. Credenciales -------------------------------------------------------
$destino = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
if (-not (Test-Path $destino)) {
    $candidatos = @(
        (Join-Path $env:USERPROFILE "Downloads\kaggle.json"),
        (Join-Path $raiz "kaggle.json")
    ) | Where-Object { Test-Path $_ }

    if (-not $candidatos) {
        throw "No encuentro kaggle.json. Descarga el token en kaggle.com/settings -> API -> Create New Token y dejalo en tu carpeta de Descargas."
    }
    New-Item -ItemType Directory -Force (Split-Path $destino) | Out-Null
    Copy-Item $candidatos[0] $destino -Force
    Write-Host "Token copiado desde $($candidatos[0]) a $destino" -ForegroundColor Green
}
# El paquete kaggle avisa si el fichero es legible por otros usuarios
icacls $destino /inheritance:r /grant:r "$($env:USERNAME):(R)" | Out-Null

# --- 2. Descarga -----------------------------------------------------------
$datos = Join-Path $raiz "data\raw\rsna"
New-Item -ItemType Directory -Force $datos | Out-Null
$zip = Join-Path $datos "rsna-pneumonia-detection-challenge.zip"

if (-not (Test-Path $zip)) {
    Write-Host "`nDescargando RSNA (~3,5 GB). Esto tarda un rato..." -ForegroundColor Cyan
    & $py -m kaggle competitions download -c rsna-pneumonia-detection-challenge -p $datos
} else {
    Write-Host "El zip ya esta descargado, salto la descarga." -ForegroundColor Yellow
}

# --- 3. Descompresion ------------------------------------------------------
if (-not (Test-Path (Join-Path $datos "stage_2_train_labels.csv"))) {
    Write-Host "`nDescomprimiendo..." -ForegroundColor Cyan
    Expand-Archive -Path $zip -DestinationPath $datos -Force
} else {
    Write-Host "Ya estaba descomprimido." -ForegroundColor Yellow
}

$n = (Get-ChildItem (Join-Path $datos "stage_2_train_images") -Filter *.dcm -ErrorAction SilentlyContinue).Count
Write-Host "`n$n imagenes DICOM en data\raw\rsna\stage_2_train_images" -ForegroundColor Green

# --- 4. Manifiesto + split por paciente -----------------------------------
Write-Host "`nConstruyendo manifiesto (lee las cabeceras DICOM, tarda unos minutos)..." -ForegroundColor Cyan
& $py -m src.prepare_data --dataset rsna --root $datos --out data\manifest_rsna.csv

Write-Host "`nListo. Siguiente paso: entrenar en Colab con notebooks/00_colab_entrenamiento.ipynb" -ForegroundColor Green
