@echo off
setlocal
cd /d "%~dp0"
set "TRAJECTORY_PYTHON="
if exist ".venv\Scripts\python.exe" set "TRAJECTORY_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined TRAJECTORY_PYTHON if exist "..\..\work\python312\python.exe" set "TRAJECTORY_PYTHON=%~dp0..\..\work\python312\python.exe"
if not defined TRAJECTORY_PYTHON goto missing_python
"%TRAJECTORY_PYTHON%" -c "import streamlit,plotly,pandas" >nul 2>nul
if errorlevel 1 (
    echo Website packages are not installed in this local interpreter yet.
    echo Run the following command in this folder, then try again:
    echo "%TRAJECTORY_PYTHON%" -m pip install -r requirements.txt
    pause
    exit /b 1
)
"%TRAJECTORY_PYTHON%" -m streamlit run app.py
goto finish
:missing_python
echo Please install Python 3.12 and the packages in requirements.txt first.
echo See README.md for setup and deployment instructions.
pause
:finish
if errorlevel 1 pause
endlocal
