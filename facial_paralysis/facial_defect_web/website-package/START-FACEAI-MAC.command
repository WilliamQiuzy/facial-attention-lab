#!/bin/zsh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if ! command -v node >/dev/null 2>&1; then
  open "https://nodejs.org/en/download"
  osascript -e 'display dialog "FaceAI needs Node.js 22 LTS. The download page has been opened. Install Node.js, then open this file again." buttons {"OK"} default button "OK" with title "FaceAI"'
  exit 1
fi

NODE_MAJOR="$(node -p "Number(process.versions.node.split('.')[0])")"
if (( NODE_MAJOR < 22 )); then
  open "https://nodejs.org/en/download"
  osascript -e 'display dialog "FaceAI needs Node.js 22 LTS or newer. Please update Node.js, then open this file again." buttons {"OK"} default button "OK" with title "FaceAI"'
  exit 1
fi

exec node "$SCRIPT_DIR/server.mjs"
