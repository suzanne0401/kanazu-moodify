@echo off
setlocal

echo [MOAI] Zatrzymywanie procesow python.exe (Flask)...
taskkill /F /IM python.exe >nul 2>&1

echo [MOAI] Gotowe.
endlocal
