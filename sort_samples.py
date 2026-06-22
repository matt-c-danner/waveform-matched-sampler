#!/usr/bin/env python3
"""
sort_samples.py
Detects pitch of audio files and copies them into per-note folders.

Usage:
  python sort_samples.py <input_dir> <output_dir> [--dry-run]

Requires:
  pip install librosa numpy
"""

import sys
import shutil
import numpy as np
from pathlib import Path

try:
    import librosa
    import soundfile as sf
except ImportError as e:
    print(f"Error: missing dependency — {e}\nRun: pip install librosa soundfile")
    sys.exit(1)

AUDIO_EXT = {'.wav', '.aif', '.aiff', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
# Formats pygame.mixer.Sound can't load — convert these to WAV on output
NEEDS_CONVERSION = {'.mp3', '.flac', '.ogg', '.m4a', '.aac'}
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_to_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{(midi // 12) - 1}"


def detect_pitch(filepath: Path):
    """Return predominant MIDI note (0-127) or None."""
    y, sr = librosa.load(str(filepath), duration=4.0, mono=True)
    y_harm = librosa.effects.harmonic(y)

    pitches, magnitudes = librosa.piptrack(y=y_harm, sr=sr, threshold=0.1)

    hits = []
    for t in range(pitches.shape[1]):
        idx = magnitudes[:, t].argmax()
        freq = pitches[idx, t]
        mag  = magnitudes[idx, t]
        if 20.0 < freq < 8000.0:
            hits.append((freq, mag))

    if not hits:
        return None

    hits.sort(key=lambda x: x[1], reverse=True)
    top_freqs = [h[0] for h in hits[:max(1, len(hits) // 3)]]
    midi = round(69 + 12 * np.log2(np.median(top_freqs) / 440.0))
    return int(midi) if 0 <= midi <= 127 else None


def sort_samples(input_dir: str, output_dir: str, dry_run: bool = False, write_coll: bool = False) -> None:
    src = Path(input_dir)
    dst = Path(output_dir)

    if not src.exists():
        print(f"Error: '{input_dir}' does not exist")
        sys.exit(1)

    files = sorted(f for f in src.rglob('*') if f.is_file() and f.suffix.lower() in AUDIO_EXT)

    if not files:
        print("No audio files found.")
        return

    print(f"Found {len(files)} audio files\n")

    note_files = {}
    undetected = []

    for i, fp in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {fp.name} ... ", end='', flush=True)
        midi = detect_pitch(fp)

        if midi is None:
            print("NO PITCH DETECTED")
            undetected.append(fp)
            continue

        name = midi_to_name(midi)
        tag = " [convert→wav]" if fp.suffix.lower() in NEEDS_CONVERSION else ""
        print(f"MIDI {midi} ({name}){tag}")

        convert = fp.suffix.lower() in NEEDS_CONVERSION
        out_suffix = '.wav' if convert else fp.suffix
        dest_name = fp.stem + out_suffix
        if not dry_run:
            folder = dst / name
            folder.mkdir(parents=True, exist_ok=True)
            dest = folder / dest_name
            n = 1
            while dest.exists():
                dest = folder / f"{fp.stem}_{n}{out_suffix}"
                n += 1
            if convert:
                y, sr = librosa.load(str(fp), sr=None, mono=False)
                # soundfile expects (samples, channels); librosa gives (channels, samples) for stereo
                if y.ndim > 1:
                    y = y.T
                sf.write(str(dest), y, sr)
            else:
                shutil.copy2(fp, dest)
            dest_name = dest.name

        note_files.setdefault(midi, []).append(dest_name)

    print(f"\n--- Summary ---")
    print(f"Sorted : {sum(len(v) for v in note_files.values())} files across {len(note_files)} notes")

    if undetected:
        print(f"Missed : {len(undetected)} files -> _undetected/")
        if not dry_run:
            u = dst / "_undetected"
            u.mkdir(parents=True, exist_ok=True)
            for fp in undetected:
                if fp.suffix.lower() in NEEDS_CONVERSION:
                    y, sr = librosa.load(str(fp), sr=None, mono=False)
                    if y.ndim > 1:
                        y = y.T
                    sf.write(str(u / (fp.stem + '.wav')), y, sr)
                else:
                    shutil.copy2(fp, u / fp.name)

    if note_files:
        print("\nNotes found:")
        for m in sorted(note_files):
            print(f"  {m:3d}  {midi_to_name(m):4s}  {len(note_files[m])} sample(s)")

    if write_coll and not dry_run and note_files:
        coll_path = dst / "samples.coll"
        with open(coll_path, 'w') as f:
            for m in sorted(note_files):
                filenames = ' '.join(note_files[m])
                f.write(f"{m}, {filenames};\n")
        print(f"\nWrote coll: {coll_path}")
        print("In Max: click 'read' on the coll object and select this file.")
        print(f"Also add '{dst}' to Max's search path: Options > File Preferences")


if __name__ == "__main__":
    argv = sys.argv[1:]
    dry = '--dry-run' in argv
    coll = '--write-coll' in argv
    argv = [a for a in argv if not a.startswith('--')]
    if len(argv) != 2:
        print("Usage: python sort_samples.py <input_dir> <output_dir> [--dry-run] [--write-coll]")
        sys.exit(1)
    sort_samples(argv[0], argv[1], dry, coll)
