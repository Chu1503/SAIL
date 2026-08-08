@echo off
REM Double-click this to start the VEINZ local pipeline: SAM2 arm isolation,
REM CUBITAL vein extraction, and the public tunnel. Three windows will open.
REM Copy this file (and this whole "windows" folder) to somewhere on your
REM Windows filesystem, e.g. the Desktop -- it launches commands inside WSL.

echo Starting vein extraction service (CUBITAL, cubital-env)...
start "VEINZ Vein Service" wsl.exe -d Ubuntu -- bash -lc "source ~/cubital-env/bin/activate && cd ~/SAIL/local-arm-mask && uvicorn vein_server:app --host 127.0.0.1 --port 8001"

echo Waiting ~15s for the vein model to load...
timeout /t 15 /nobreak >nul

echo Starting the main server (SAM2, veinz-env)...
start "VEINZ Server" wsl.exe -d Ubuntu -- bash -lc "source ~/veinz-env/bin/activate && cd ~/SAIL/local-arm-mask && uvicorn server:app --host 0.0.0.0 --port 8000"

echo Waiting ~30s for SAM2 to load onto the GPU...
timeout /t 30 /nobreak >nul

echo Starting the ngrok tunnel on your static domain...
start "VEINZ Tunnel" wsl.exe -d Ubuntu -- bash -lc "ngrok http --url https://stubbly-footnote-scrutiny.ngrok-free.dev 8000"

echo Waiting ~5s for the tunnel to come up...
timeout /t 5 /nobreak >nul

echo Checking that the public URL is responding...
wsl.exe -d Ubuntu -- curl -s https://stubbly-footnote-scrutiny.ngrok-free.dev/health
echo.
echo.
echo If you see {"status":"ok",...} above, everything is running.
echo Keep all three new windows open -- closing any one of them stops that piece.
echo.
pause
