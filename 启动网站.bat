@echo off
setlocal
cd /d "%~dp0"
if exist "..\..\work\python312\python.exe" (
    "..\..\work\python312\python.exe" -m streamlit run app.py
    goto finish
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3.12 -m streamlit run app.py
    goto finish
)
echo Please install Python 3.12 and the packages in requirements.txt first.
echo See README.md for setup and deployment instructions.
:finish
if errorlevel 1 pause
endlocal
