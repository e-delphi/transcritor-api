# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo
#
# Sobe o servidor audio.cpp (GPU) e a API. Ctrl+C encerra os dois.
#
#   iniciar.bat [-Porta 8000]   (ou: powershell -ExecutionPolicy Bypass -File start.ps1)

param(
    [string]$Destino = (Join-Path $env:LOCALAPPDATA "TranscritorAPI"),
    [int]$Porta = 8000,
    [string]$Endereco = "127.0.0.1"
)

# "Continue" de propósito: no PowerShell 5.1, com "Stop", qualquer linha que um
# programa escreva no stderr (logs do uvicorn, a lista de GPUs do audio.cpp) vira
# erro fatal. Os programas são conferidos pelo código de saída; os cmdlets
# críticos usam -ErrorAction Stop.
$ErrorActionPreference = "Continue"
if (-not (Test-Path "$Destino\server.json")) {
    throw "instalação não encontrada em $Destino; rode primeiro instalar.bat"
}
$Repo = (Get-Content -Raw "$Destino\repo.txt").Trim()

# Se algo já responde na 8081, o teste de saúde abaixo passaria com o servidor de
# outra pessoa, e a API usaria modelos e configuração que não são os desta instalação.
try {
    Invoke-RestMethod -TimeoutSec 2 "http://127.0.0.1:8081/health" | Out-Null
    throw "a porta 8081 já está em uso por outro servidor; encerre-o antes de iniciar"
} catch [System.Net.WebException] { }

$servidor = Start-Process -PassThru -NoNewWindow `
    -FilePath "$Destino\audiocpp\audiocpp_server.exe" `
    -ArgumentList @("--config", "`"$Destino\server.json`"") `
    -RedirectStandardOutput "$Destino\logs\audiocpp.log" `
    -RedirectStandardError "$Destino\logs\audiocpp.err.log"

try {
    Write-Host "audio.cpp iniciando (PID $($servidor.Id))..."
    $ok = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        if ($servidor.HasExited) { break }
        try { Invoke-RestMethod -TimeoutSec 2 "http://127.0.0.1:8081/health" | Out-Null; $ok = $true; break } catch { }
    }
    if (-not $ok) {
        Get-Content -Tail 20 "$Destino\logs\audiocpp.err.log", "$Destino\logs\audiocpp.log" -ErrorAction SilentlyContinue
        throw "o servidor audio.cpp não subiu; veja os logs em $Destino\logs"
    }

    $env:AUDIOCPP_URL = "http://127.0.0.1:8081"
    $env:JOBS_DIR = "$Destino\data\jobs"
    if (-not $env:JOB_TTL_SECONDS) { $env:JOB_TTL_SECONDS = "21600" }
    if (-not $env:MAX_UPLOAD_MB) { $env:MAX_UPLOAD_MB = "1024" }

    Write-Host "API em http://${Endereco}:$Porta  (documentação em /docs)"
    & "$Destino\venv\Scripts\python.exe" -m uvicorn app.main:app --app-dir $Repo `
        --host $Endereco --port $Porta --timeout-keep-alive 300
}
finally {
    if (-not $servidor.HasExited) {
        Stop-Process -Id $servidor.Id -Force
        Write-Host "audio.cpp encerrado."
    }
}
