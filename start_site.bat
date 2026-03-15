@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [MOAI] Tworzenie virtual environment...
  py -m venv .venv
)

echo [MOAI] Instalowanie zaleznosci...
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo [MOAI] Otwieranie strony w przegladarce: http://localhost:8000
start "" "http://localhost:8000"

echo [MOAI] Uruchamianie serwera Flask...
".venv\Scripts\python.exe" app.py

endlocal
