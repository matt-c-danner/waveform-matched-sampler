#!/usr/bin/env python3
"""
decompose_sample.py
Splits a long audio file into individual samples at detected note onsets.
Optionally detects pitch and organises slices into per-note subfolders
ready for the Waveform-Matched Sampler.

Usage:
  python3 decompose_sample.py <input_file> <output_dir> [options]

Options:
  --min-duration SECS   Discard slices shorter than this  (default: 0.1)
  --max-duration SECS   Split slices longer than this     (default: 4.0)
  --sort                Detect pitch and write to note-named subfolders
  --dry-run             Preview slices without writing any files
  --prefix NAME         Filename prefix (default: input file stem)

Requires:
  pip install librosa soundfile numpy
  brew install ffmpeg   (needed for m4a / AAC decoding)
"""

import sys
import argparse
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import librosa
    import soundfile as sf
except ImportError as e:
    print(f"Error: missing dependency — {e}\nRun: pip install librosa soundfile")
    sys.exit(1)

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_MAP = {
    'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11,
    'C#': 1, 'D#': 3, 'F#': 6, 'G#': 8, 'A#': 10,
}


def midi_to_name(midi):
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def detect_pitch(y, sr):
    """Return predominant MIDI note (0-127) or None."""
    y_harm = librosa.effects.harmonic(y)
    pitches, magnitudes = librosa.piptrack(y=y_harm, sr=sr, threshold=0.1)
    hits = []
    for t in range(pitches.shape[1]):
        idx = magnitudes[:, t].argmax()
        freq = pitches[idx, t]
        mag = magnitudes[idx, t]
        if 20.0 < freq < 8000.0:
            hits.append((freq, mag))
    if not hits:
        return None
    hits.sort(key=lambda x: x[1], reverse=True)
    top = [h[0] for h in hits[:max(1, len(hits) // 3)]]
    midi = round(69 + 12 * np.log2(np.median(top) / 440.0))
    return int(midi) if 0 <= midi <= 127 else None


def fade_edges(audio, sr, fade_in=0.005, fade_out=0.03):
    """Apply short fade-in and fade-out to avoid clicks."""
    n_in  = min(int(sr * fade_in),  len(audio) // 4)
    n_out = min(int(sr * fade_out), len(audio) // 4)
    result = audio.copy()
    if n_in > 0:
        result[:n_in]  *= np.linspace(0.0, 1.0, n_in)
    if n_out > 0:
        result[-n_out:] *= np.linspace(1.0, 0.0, n_out)
    return result


def find_slices(y, sr, min_dur, max_dur):
    """
    Return list of (start_sample, end_sample) tuples.
    Uses onset detection, then splits any segment longer than max_dur
    into equal-length chunks.
    """
    # Onset detection with backtracking to actual attack
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, backtrack=True,
        pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.07, wait=10
    )
    onset_samples = librosa.frames_to_samples(onset_frames).tolist()

    # Build boundaries: 0 ... onsets ... end
    boundaries = sorted(set([0] + onset_samples + [len(y)]))

    min_samp = int(min_dur * sr)
    max_samp = int(max_dur * sr)

    raw_slices = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end - start < min_samp:
            continue
        # Split segments that exceed max_dur into equal chunks
        length = end - start
        if length > max_samp:
            n_chunks = int(np.ceil(length / max_samp))
            chunk_len = length // n_chunks
            for c in range(n_chunks):
                cs = start + c * chunk_len
                ce = cs + chunk_len if c < n_chunks - 1 else end
                if ce - cs >= min_samp:
                    raw_slices.append((cs, ce))
        else:
            raw_slices.append((start, end))

    return raw_slices


def decompose(input_file, output_dir, min_dur=0.1, max_dur=4.0,
              do_sort=False, dry_run=False, prefix=None):
    SUPPORTED = {'.wav', '.aif', '.aiff', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    src = Path(input_file)
    dst = Path(output_dir)

    if not src.exists():
        print(f"Error: '{input_file}' does not exist")
        sys.exit(1)
    if src.suffix.lower() not in SUPPORTED:
        print(f"Warning: '{src.suffix}' may not be supported. Trying anyway…")

    print(f"Loading {src.name} …")
    y, sr = librosa.load(str(src), sr=None, mono=True)
    duration = len(y) / sr
    print(f"  Duration : {duration:.2f}s   Sample rate: {sr} Hz")

    print("Detecting onsets …")
    slices = find_slices(y, sr, min_dur, max_dur)
    print(f"  Found {len(slices)} slice(s)\n")

    if not slices:
        print("No slices found — try lowering --min-duration.")
        return

    stem = prefix or src.stem
    note_files = {}
    undetected = []

    for i, (start, end) in enumerate(slices, 1):
        seg = y[start:end]
        dur = len(seg) / sr
        filename = f"{stem}_{i:03d}.wav"
        print(f"[{i}/{len(slices)}] {filename}  ({dur:.2f}s)", end='', flush=True)

        midi = None
        if do_sort:
            midi = detect_pitch(seg, sr)
            if midi is not None:
                print(f"  →  {midi_to_name(midi)} ({midi})", end='')
            else:
                print("  →  no pitch", end='')

        print()

        if dry_run:
            continue

        if do_sort:
            if midi is not None:
                folder = dst / midi_to_name(midi)
                note_files.setdefault(midi, []).append(filename)
            else:
                folder = dst / "_undetected"
                undetected.append(filename)
        else:
            folder = dst

        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / filename
        n = 1
        while dest.exists():
            dest = folder / f"{stem}_{i:03d}_{n}.wav"
            n += 1

        # Load the slice from source preserving original channel count
        y_slice, _ = librosa.load(str(src), sr=sr, mono=False,
                                   offset=start / sr, duration=(end - start) / sr)
        if y_slice.ndim == 1:
            # Mono
            sf.write(str(dest), fade_edges(y_slice, sr), sr)
        else:
            # Stereo: (channels, samples) → apply fade per channel → (samples, channels)
            faded = np.stack([fade_edges(y_slice[ch], sr)
                              for ch in range(y_slice.shape[0])], axis=1)
            sf.write(str(dest), faded, sr)

    if dry_run:
        print("\n(dry run — no files written)")
        return

    print(f"\n--- Summary ---")
    print(f"Written : {len(slices) - len(undetected)} slice(s) to '{dst}'")
    if do_sort:
        if undetected:
            print(f"No pitch : {len(undetected)} slice(s) → _undetected/")
        if note_files:
            print("\nNotes found:")
            for m in sorted(note_files):
                print(f"  {m:3d}  {midi_to_name(m):4s}  {len(note_files[m])} sample(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompose a long sample into slices.")
    parser.add_argument("input_file",  help="Source audio file")
    parser.add_argument("output_dir",  help="Destination folder")
    parser.add_argument("--min-duration", type=float, default=0.1,
                        metavar="SECS", help="Minimum slice duration (default 0.1)")
    parser.add_argument("--max-duration", type=float, default=4.0,
                        metavar="SECS", help="Maximum slice duration (default 4.0)")
    parser.add_argument("--sort",     action="store_true",
                        help="Detect pitch and sort into note subfolders")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Preview only, write nothing")
    parser.add_argument("--prefix",   default=None,
                        help="Output filename prefix (default: input stem)")
    args = parser.parse_args()

    decompose(
        args.input_file, args.output_dir,
        min_dur=args.min_duration, max_dur=args.max_duration,
        do_sort=args.sort, dry_run=args.dry_run, prefix=args.prefix,
    )
