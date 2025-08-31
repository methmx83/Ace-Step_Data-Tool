@echo off
set "CONDA_ROOT=Z:\AI\software\miniconda3"
set "ENV_NAME=acedata"
set "WORKDIR=Z:\AI\projects\music\DATA_TOOL\scripts\ui"

start "DATA-TOOL V2" cmd /k ""%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME% && cd /d %WORKDIR% && python ui.py"