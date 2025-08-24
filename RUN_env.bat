@echo off
set "CONDA_ROOT=Z:\AI\software\miniconda3"
set "ENV_NAME=ace-data_v2_env"
set "WORKDIR=Z:\AI\projects\music\data_v2"

start "ACE-STEP DATA-TOOL" cmd /k ""%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME% && cd /d %WORKDIR% && echo Environment %ENV_NAME% activated in %WORKDIR%"