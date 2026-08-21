@echo off
setlocal
cd /d "%~dp0"

where node >nul 2>&1
if errorlevel 1 (
  start "" "https://nodejs.org/en/download"
  echo FaceAI needs Node.js 22 LTS. Install Node.js, then open this file again.
  pause
  exit /b 1
)

for /f %%V in ('node -p "Number(process.versions.node.split('.')[0])"') do set NODE_MAJOR=%%V
if %NODE_MAJOR% LSS 22 (
  start "" "https://nodejs.org/en/download"
  echo FaceAI needs Node.js 22 LTS or newer. Update Node.js, then open this file again.
  pause
  exit /b 1
)

node "%~dp0server.mjs"
if errorlevel 1 pause
