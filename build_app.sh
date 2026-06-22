#!/bin/bash
# Build waveform_matched_sampler.app for macOS distribution
set -e

PYTHON=/opt/homebrew/bin/python3.14

echo "==> Installing dependencies..."
$PYTHON -m pip install --break-system-packages -r requirements.txt
$PYTHON -m pip install --break-system-packages librosa soundfile python-rtmidi pyinstaller pillow

echo "==> Generating app icon..."
$PYTHON make_icon.py

echo "==> Building .app..."
$PYTHON -m PyInstaller \
  --onedir \
  --windowed \
  --name "Waveform Matched Sampler" \
  --osx-bundle-identifier com.local.waveformsampler \
  --icon AppIcon.icns \
  --hidden-import=pygame.pypm \
  --collect-all=pygame \
  --collect-all=librosa \
  --collect-all=soundfile \
  --collect-all=audioread \
  --collect-all=rtmidi \
  --exclude-module numba \
  --exclude-module llvmlite \
  -y \
  waveform_matched_sampler.py

echo "==> Adding microphone permission to Info.plist..."
/usr/libexec/PlistBuddy -c \
  "Add :NSMicrophoneUsageDescription string 'Required for pitch detection from audio input'" \
  "dist/Waveform Matched Sampler.app/Contents/Info.plist"

echo "==> Re-signing with entitlements..."
codesign --deep --force --sign - \
  --entitlements entitlements.plist \
  "dist/Waveform Matched Sampler.app"

echo ""
echo "Done! Find your app at:  dist/Waveform Matched Sampler.app"
echo ""
echo "To distribute: zip the .app or drag it to /Applications."
echo "Recipient needs no Python, Max, or Ableton — just the .app."
