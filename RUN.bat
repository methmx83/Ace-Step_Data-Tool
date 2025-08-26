@echo off
set "CONDA_ROOT=Z:\AI\software\miniconda3"
set "ENV_NAME=ace-data_v2_env"
set "WORKDIR=Z:\AI\projects\music\data_v2\scripts\ui"

start "DATA V2" cmd /k ""%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME% && cd /d %WORKDIR% && python app.py"