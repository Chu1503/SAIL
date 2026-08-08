@echo off
REM Double-click this to start the VEINZ local arm-isolation server + tunnel.
REM Copy this file (and this whole "windows" folder) to somewhere on your
REM Windows filesystem, e.g. the Desktop -- it launches commands inside WSL.

echo Starting VEINZ local server (SAM2 + CUBITAL, this window will show its logs)...
start "VEINZ Server" wsl.exe -d Ubuntu -- bash -lc "source ~/veinz-env/bin/activate && cd ~/SAIL/local-arm-mask && uvicorn server:app --host 0.0.0.0 --port 8000"

echo Waiting ~40s for both models to finish loading onto the GPU...
timeout /t 40 /nobreak >nul

echo Starting the ngrok tunnel on your static domain...
start "VEINZ Tunnel" wsl.exe -d Ubuntu -- bash -lc "ngrok http --url https://stubbly-footnote-scrutiny.ngrok-free.dev 8000"

echo Waiting ~5s for the tunnel to come up...
timeout /t 5 /nobreak >nul

echo Checking that the public URL is responding...
wsl.exe -d Ubuntu -- curl -s https://stubbly-footnote-scrutiny.ngrok-free.dev/health
echo.
echo.
echo If you see {"status":"ok",...} above, everything is running.
echo Keep both new windows open -- closing either one stops that piece.
echo.
pause
