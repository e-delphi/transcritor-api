@echo off
rem SPDX-License-Identifier: GPL-3.0-or-later
rem Sobe o audio.cpp e a API. Feche a janela ou use Ctrl+C para encerrar.
rem Aceita os mesmos parametros do start.ps1, por exemplo:
rem   iniciar.bat -Porta 8010
title Transcritor API
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
set CODIGO=%ERRORLEVEL%
if %CODIGO% neq 0 pause
exit /b %CODIGO%
