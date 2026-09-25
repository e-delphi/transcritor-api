# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo
#
# Instala o Transcritor API nativo no Windows, com o motor audio.cpp na GPU (Vulkan).
# Funciona com placas AMD, NVIDIA e Intel que tenham driver Vulkan. Nada disso
# precisa de Docker: o Docker Desktop não repassa GPUs AMD para containers.
#
#   powershell -ExecutionPolicy Bypass -File native\windows\install.ps1
#
# Parâmetros:
#   -Destino     pasta da instalação (padrão: %LOCALAPPDATA%\TranscritorAPI)
#   -ModelosDe   pasta com os .gguf já baixados, para não baixar de novo
#
# Todo download é conferido pelo SHA-256 publicado; se não bater, a instalação para.

param(
    [string]$Destino = (Join-Path $env:LOCALAPPDATA "TranscritorAPI"),
    [string]$ModelosDe = ""
)

# "Continue" de propósito: no PowerShell 5.1, com "Stop", qualquer linha que um
# programa escreva no stderr (logs do uvicorn, a lista de GPUs do audio.cpp) vira
# erro fatal. Os programas são conferidos pelo código de saída; os cmdlets
# críticos usam -ErrorAction Stop.
$ErrorActionPreference = "Continue"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

$AudioCppVersao = "v0.8.1"
$AudioCppZip = "audio-$AudioCppVersao-bin-windows-x64-vulkan.zip"
$AudioCppUrl = "https://github.com/0xShug0/audio.cpp/releases/download/$AudioCppVersao/$AudioCppZip"
$AudioCppSha = "c787971e025ba8ef900f0482a2cc36a049367081fe89f4841aae521a0b49de32"

$HF = "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main"
$Modelos = @(
    @{ Nome = "qwen3-asr-1.7b-q8_0.gguf"; Caminho = "Qwen3-ASR-1.7B-GGUF";
       Sha = "da4fc2ac7f24dee784d1684eb1f35836cdbf559519452ae11777670734c0a4f8" },
    @{ Nome = "qwen3-forced-aligner-0.6b-q8_0.gguf"; Caminho = "Qwen3-ForcedAligner-0.6B-GGUF";
       Sha = "75209490b11cec2b0db749ca5f4ff92266f58efd30f7fd04d9eb2a3ac9cc929f" }
)

function Passo($texto) { Write-Host "`n==> $texto" -ForegroundColor Cyan }

function Conferir($arquivo, $esperado) {
    $hash = (Get-FileHash -Algorithm SHA256 -ErrorAction Stop $arquivo).Hash.ToLower()
    if ($hash -ne $esperado) {
        throw "SHA-256 de $(Split-Path $arquivo -Leaf) não confere (obtido $hash, esperado $esperado). Apague o arquivo e rode de novo."
    }
}

function Baixar($url, $arquivo) {
    # curl.exe vem com o Windows 10+ e retoma downloads interrompidos (-C -).
    & curl.exe -L --fail --retry 3 -C - -o $arquivo $url
    if ($LASTEXITCODE -ne 0) { throw "falha ao baixar $url" }
}

New-Item -ItemType Directory -Force -ErrorAction Stop -Path $Destino, "$Destino\models", "$Destino\data\jobs", "$Destino\logs" | Out-Null

# ---------------------------------------------------------------- Python ----
Passo "Python 3.11 ou mais novo"
$pyExe = $null
foreach ($cand in @("py -3.13", "py -3.12", "py -3.11", "python")) {
    $partes = $cand.Split(" ")
    $exe = $partes[0]
    $extra = @($partes | Select-Object -Skip 1)
    try {
        $v = & $exe @extra -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v -and [version]$v -ge [version]"3.11") {
            $pyExe = $exe; $pyExtra = $extra; $pyVersao = $v; break
        }
    } catch { }
}
if (-not $pyExe) { throw "Python 3.11+ não encontrado. Instale em https://www.python.org/downloads/ e rode de novo." }
Write-Host "usando: $pyExe $($pyExtra -join ' ') (Python $pyVersao)"

Passo "Ambiente isolado em $Destino\venv"
if (-not (Test-Path "$Destino\venv\Scripts\python.exe")) {
    & $pyExe @pyExtra -m venv "$Destino\venv"
    if ($LASTEXITCODE -ne 0) { throw "falha ao criar o ambiente isolado" }
}
& "$Destino\venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r "$PSScriptRoot\requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "falha ao instalar as dependências Python" }

# -------------------------------------------------------------- audio.cpp ---
Passo "audio.cpp $AudioCppVersao (Vulkan)"
if (-not (Test-Path "$Destino\audiocpp\audiocpp_server.exe")) {
    $zip = Join-Path $Destino $AudioCppZip
    if (-not (Test-Path $zip)) { Baixar $AudioCppUrl $zip }
    Conferir $zip $AudioCppSha
    Expand-Archive -Force -ErrorAction Stop $zip "$Destino\audiocpp"
    Remove-Item -ErrorAction Stop $zip
}
& "$Destino\audiocpp\audiocpp_cli.exe" --list-devices 2>$null | Select-String "Vulkan:|CPU:" | ForEach-Object { Write-Host "  $_" }

# ---------------------------------------------------------------- modelos ---
Passo "Modelos Qwen3-ASR e Qwen3-ForcedAligner (~3,6 GB)"
foreach ($m in $Modelos) {
    $alvo = Join-Path "$Destino\models" $m.Nome
    if (Test-Path $alvo) {
        Write-Host "  $($m.Nome): já presente"
    } elseif ($ModelosDe -and (Test-Path (Join-Path $ModelosDe $m.Nome))) {
        $origem = Join-Path $ModelosDe $m.Nome
        try {
            New-Item -ItemType HardLink -ErrorAction Stop -Path $alvo -Target $origem | Out-Null
            Write-Host "  $($m.Nome): vinculado de $ModelosDe (sem cópia)"
        } catch {
            Copy-Item -ErrorAction Stop $origem $alvo
            Write-Host "  $($m.Nome): copiado de $ModelosDe"
        }
    } else {
        Write-Host "  $($m.Nome): baixando"
        Baixar "$HF/$($m.Caminho)/$($m.Nome)" $alvo
    }
    Conferir $alvo $m.Sha
}

# ---------------------------------------------------------- configuração ----
Passo "Configuração"
$servidor = @{
    host = "127.0.0.1"; port = 8081; backend = "vulkan"; device = 0; threads = 8
    lazy_load = $true; max_loaded_models = 2; idle_unload_ms = 0; min_free_memory_mb = 0
    models = @(
        @{ id = "qwen3-asr"; family = "qwen3_asr"; task = "asr"; mode = "offline"
           path = (Join-Path "$Destino\models" "qwen3-asr-1.7b-q8_0.gguf") -replace "\\", "/" },
        @{ id = "qwen3-align"; family = "qwen3_forced_aligner"; task = "align"; mode = "offline"
           path = (Join-Path "$Destino\models" "qwen3-forced-aligner-0.6b-q8_0.gguf") -replace "\\", "/" }
    )
}
$utf8 = New-Object System.Text.UTF8Encoding $false
[IO.File]::WriteAllText("$Destino\server.json", ($servidor | ConvertTo-Json -Depth 5), $utf8)
[IO.File]::WriteAllText("$Destino\repo.txt", $Repo, $utf8)

Passo "Pronto"
Write-Host "Instalado em $Destino"
Write-Host "Para iniciar:  $PSScriptRoot\iniciar.bat"
