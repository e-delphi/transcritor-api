@echo off
rem SPDX-License-Identifier: GPL-3.0-or-later
rem Instala o Transcritor API nativo (GPU via Vulkan). Pode abrir com duplo clique.
rem Aceita os mesmos parametros do install.ps1, por exemplo:
rem   instalar.bat -ModelosDe C:\modelos
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set CODIGO=%ERRORLEVEL%
echo.
if %CODIGO% neq 0 (echo A instalacao falhou. Veja a mensagem acima.) else (echo Instalacao concluida. Para iniciar, abra iniciar.bat)
pause
exit /b %CODIGO%
