# Waveform-Matched Sampler

A standalone macOS app that listens to a live audio input, detects the played pitch in real time, and fires a matching sample from your library. Four rotating voices let samples overlap while a note is held.

## How it works

The instrument has three input modes — all three share the same sample selection, retrigger, and voice engine:

**Audio mode** — the default. Audio from the selected input is captured in 2048-sample frames at 44.1 kHz. A pure-NumPy implementation of the **YIN algorithm** estimates the fundamental frequency of each frame, converts it to the nearest MIDI note, and triggers a matching sample. While a pitch is detected continuously, the sample retriggers on a configurable interval.

**MIDI mode** — connects to any CoreMIDI/ALSA input device. Note-on messages trigger a sample; note-off silences the retrigger for that note.

**Keyboard mode** — the computer keyboard acts as a two-row piano layout. Pressing a key triggers a sample and holds it (retrigger fires while the key is held); releasing the key silences it. `z`/`x` shift the base octave down/up.

In all modes:
- A sample keyed to the exact MIDI note is played, or the nearest available note within one octave.
- When multiple samples exist for a note, one is chosen at random on every trigger and retrigger.
- Playback is spread across **4 rotating voices** so new samples overlap previous ones.

## Requirements

- macOS (built and distributed as a `.app`)
- Python 3.9+ (only needed if running from source)
- `sounddevice >= 0.4.6`
- `pygame >= 2.5.0`
- `numpy >= 1.24.0`

## Installation

### Pre-built app

Drag `Waveform Matched Sampler.app` from the `dist/` folder into `/Applications`. No Python or additional tools required.

### From source

```bash
pip install -r requirements.txt
python waveform_matched_sampler.py
```

### Build the app yourself

```bash
bash build_app.sh
# Output: dist/Waveform Matched Sampler.app
```

## Preparing samples with sort_samples.py

`sort_samples.py` automatically detects the pitch of a folder of audio files and copies them into the subfolder layout the app expects.

### Requirements

```bash
pip3 install librosa numpy
```

### Usage

```bash
python3 sort_samples.py <input_dir> <output_dir> [--dry-run]
```

Run with `--dry-run` first to preview what pitch gets detected for each file without copying anything.

### Example

```bash
python3 sort_samples.py ~/Downloads/raw_samples ~/Music/sorted_samples --dry-run
python3 sort_samples.py ~/Downloads/raw_samples ~/Music/sorted_samples
```

Output structure:

```
sorted_samples/
  C4/
    piano_c4.wav
  F#3/
    cello_fs3.wav
  _undetected/       ← files where no pitch could be detected
    noise_pad.wav
```

Then point the app's **Sample Folder** browser at `sorted_samples/`.

**Supported input formats:** `.wav`, `.aif`, `.aiff`, `.mp3`, `.flac`, `.ogg`

Files where no pitch can be detected are copied to `_undetected/` for manual review.

## Sample library layout

The app accepts two folder layouts (and will mix both inside the same folder):

### Subfolders named by note

```
my_samples/
  C4/        # or  60/
    violin_long.wav
    violin_short.wav
  F#3/       # or  54/
    cello.wav
```

### Flat files with a note-encoded stem

The stem before the first `_` (or the entire stem if no `_` is present) is parsed as a note name or MIDI number.

```
my_samples/
  60.wav
  C4_violin.wav
  F#3_cello_pizz.wav
  Bb5.aif
```

**Supported formats:** `.wav`, `.aif`, `.aiff`

**Note name syntax:** `C4`, `F#3`, `Bb5`, `60` (MIDI number), etc. Sharps use `#`, flats use `b`.

## UI reference

| Control | Description |
|---|---|
| **Note display** | Shows the detected/played note name and MIDI number (`C4 (60)`). Dash when silent. |
| **Voice LEDs** | Flash blue when a voice fires. Four LEDs, one per playback voice. |
| **Sample Folder** | Browse button opens a folder picker. The count of loaded samples is shown below. |
| **Input Mode** | Radio buttons: Audio / MIDI / Keyboard. Switches the input config section below. Disabled while running. |
| **Audio Input** | (Audio mode) Drop-down list of all available audio input devices. |
| **MIDI Input** | (MIDI mode) Drop-down list of all MIDI input devices detected at launch. |
| **Keyboard** | (Keyboard mode) Displays the key layout and the current base note. `z`/`x` shift the octave while running. |
| **Retrigger Interval** | How often (in ms) the current note re-fires a new random sample while held. Range: 100 – 8000 ms, default 2000 ms. |
| **Start / Stop** | Starts or stops the selected input. Green = stopped, red = running. |

## Keyboard layout

The keyboard mode uses the standard "musical typing" layout:

```
  W  E     T  Y  U     O  P     ← sharps / flats
A  S  D  F  G  H  J  K  L       ← naturals

z = octave ↓    x = octave ↑
```

With the default base note of C4:

| Key | Note | MIDI |
|---|---|---|
| A | C4 | 60 |
| W | C#4 | 61 |
| S | D4 | 62 |
| E | D#4 | 63 |
| D | E4 | 64 |
| F | F4 | 65 |
| T | F#4 | 66 |
| G | G4 | 67 |
| Y | G#4 | 68 |
| H | A4 | 69 |
| U | A#4 | 70 |
| J | B4 | 71 |
| K | C5 | 72 |
| O | C#5 | 73 |
| L | D5 | 74 |
| P | D#5 | 75 |

OS key-repeat events are filtered — each key triggers exactly once until released.

## Pitch detection details (Audio mode)

- **Algorithm:** YIN with parabolic interpolation for sub-sample accuracy.
- **Silence gate:** Frames with RMS below 0.005 are skipped entirely — no false triggers on noise.
- **Confidence threshold:** A detected pitch is only accepted when the YIN confidence score exceeds 0.7.
- **Detection range:** ~20 Hz and above; MIDI notes 0–127 (C-1 through G9).

## Sample selection

If the detected MIDI note has no exact match in the library, the app searches for the **nearest loaded note**. A sample is played only if the closest match is within **one octave (12 semitones)**; otherwise playback is skipped for that frame.

When multiple samples exist for a note, one is chosen **at random** on every trigger and retrigger.

## Retrigger behaviour

While a pitch is continuously detected, a timer fires at the configured interval and replays a fresh random sample on the next available voice. The timer resets whenever the pitch changes and cancels on silence.

## Voices

Four pygame mixer channels (voices 0–3) are used in a round-robin. This means up to four samples can play simultaneously before the oldest is interrupted. The voice LEDs give a visual indication of which channel is active.

## Troubleshooting

**"Missing Dependencies" dialog on launch**
Run `pip install sounddevice pygame numpy` and relaunch.

**No audio devices listed**
Make sure macOS has granted microphone access to the Terminal (or the `.app`). Check System Settings → Privacy & Security → Microphone.

**No MIDI devices listed**
Plug in (or connect via Bluetooth) your MIDI controller before launching the app. MIDI devices are enumerated once at startup; relaunch after connecting a new device.

**Keyboard mode: keys don't respond**
Click the app window to ensure it has focus — keyboard bindings are local to the window.

**Samples not loading**
Verify file names follow the note-name format above. Only `.wav`, `.aif`, and `.aiff` files are scanned.

**Pitch detection is unreliable (Audio mode)**
- Use a clean, monophonic source (single instrument, no reverb tails bleeding into the next note).
- Lower the retrigger interval if you want faster re-sampling.
- Ensure your input level is not too quiet (RMS gate at 0.005 linear).
