#!/usr/bin/env python3
"""
Waveform-Matched Sampler
Detects pitch from audio input and plays a random matching sample.
4 rotating voices allow samples to overlap while a note is held.
"""

import os
os.environ['NUMBA_DISABLE_JIT'] = '1'  # must be set before librosa/numba import

import tkinter as tk
from tkinter import filedialog, ttk
import threading
import time
import shutil
import random
import re
import json
from pathlib import Path

PREFS_PATH = Path.home() / '.waveform_matched_sampler.json'
DEFAULT_BANK_PATH = (Path.home() / 'Library' / 'Application Support'
                     / 'WaveformMatchedSampler' / 'samples')

try:
    import sounddevice as sd
    import numpy as np
    import pygame
    import pygame.sndarray
    DEPS_OK = True
    MISSING = None
except ImportError as e:
    DEPS_OK = False
    MISSING = str(e)

MIDI_OK = False
if DEPS_OK:
    try:
        import pygame.midi
        MIDI_OK = True
    except ImportError:
        pass

RTMIDI_OK = False
try:
    import rtmidi
    RTMIDI_OK = True
except ImportError:
    pass

LIBROSA_OK = False
try:
    import librosa
    import soundfile as sf
    LIBROSA_OK = True
except ImportError:
    pass

SAMPLE_RATE = 44100
HOP_SIZE = 1024
BUFFER_SIZE = 2048

# ── Pure-NumPy YIN pitch detector ────────────────────────────────────────────

def _yin_pitch(frame, sr, threshold=0.15):
    """Return (frequency_hz, confidence) using the YIN algorithm."""
    N = len(frame)
    half = N // 2

    win = np.fft.rfft(frame, n=N * 2)
    acf = np.fft.irfft(win * np.conj(win))[:half]
    acf = acf.real
    d = acf[0] + acf[0] - 2 * acf

    cmndf = np.empty(half)
    cmndf[0] = 1.0
    running = 0.0
    for tau in range(1, half):
        running += d[tau]
        cmndf[tau] = d[tau] * tau / running if running > 0 else 1.0

    tau_est = -1
    for tau in range(2, half):
        if cmndf[tau] < threshold:
            while tau + 1 < half and cmndf[tau + 1] < cmndf[tau]:
                tau += 1
            tau_est = tau
            break

    if tau_est < 2:
        return 0.0, 0.0

    if tau_est > 0 and tau_est < half - 1:
        s0, s1, s2 = cmndf[tau_est - 1], cmndf[tau_est], cmndf[tau_est + 1]
        denom = s0 - 2 * s1 + s2
        tau_frac = tau_est + (s0 - s2) / (2 * denom) if denom != 0 else tau_est
    else:
        tau_frac = tau_est

    freq = sr / tau_frac
    confidence = 1.0 - cmndf[tau_est]
    return freq, confidence


NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

NOTE_MAP = {
    'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11,
    'CB': -1, 'DB': 1, 'EB': 3, 'FB': 4, 'GB': 6, 'AB': 8, 'BB': 10,
    'C#': 1, 'D#': 3, 'E#': 5, 'F#': 6, 'G#': 8, 'A#': 10, 'B#': 12,
}


def midi_to_name(midi):
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def name_to_midi(name):
    """Parse MIDI note number or note name like C4, F#3, Bb5."""
    name = name.strip()
    try:
        v = int(name)
        return v if 0 <= v <= 127 else None
    except ValueError:
        pass
    m = re.match(r'^([A-Ga-g][#b]?)(-?\d+)$', name)
    if m:
        note = m.group(1).upper().replace('b', 'B')
        octave = int(m.group(2))
        if note in NOTE_MAP:
            midi = (octave + 1) * 12 + NOTE_MAP[note]
            return midi if 0 <= midi <= 127 else None
    return None


# ── Keyboard layout ───────────────────────────────────────────────────────────
_KB_OFFSETS = {
    'a':  0,  'w':  1,  's':  2,  'e':  3,  'd':  4,
    'f':  5,  't':  6,  'g':  7,  'y':  8,  'h':  9,
    'u': 10,  'j': 11,  'k': 12,  'o': 13,  'l': 14,  'p': 15,
}

_KB_HINT = (
    "  W  E     T  Y  U     O  P\n"
    "A  S  D  F  G  H  J  K  L\n"
    "\n"
    "z = octave ↓    x = octave ↑"
)

# ── Colour palette ────────────────────────────────────────────────────────────
BG      = "#1e1e2e"
BG_DARK = "#11111b"
FG      = "#cdd6f4"
FG_DIM  = "#7f849c"
FG_DIMR = "#585b70"
ACCENT  = "#89b4fa"
GREEN   = "#a6e3a1"
SURFACE = "#313244"
SURF2   = "#45475a"


def _short_path(path):
    p = str(path)
    home = str(Path.home())
    if p.startswith(home):
        p = '~' + p[len(home):]
    if len(p) > 42:
        p = '…' + p[-40:]
    return p


class SamplerApp:
    NUM_VOICES = 4

    def __init__(self, root):
        self.root = root
        self.root.title("Waveform-Matched Sampler")

        self.sample_db = {}
        self._sound_names = {}          # id(Sound) → filename
        self._fx_cache = {}             # (sound_id, reverb, delay_ms, feedback) → Sound
        self._reverb_ir = None
        self._reverb_ir_ffts = {}       # fft_len → rfft(ir, n=fft_len)
        self.reverb_wet = 0.0
        self.delay_ms = 0
        self.delay_feedback = 0.5
        self._voice_samples = ['—'] * self.NUM_VOICES
        self.current_note = -1
        self.voice_idx = 0
        self.retrigger_ms = 2000
        self.running = False
        self.stream = None
        self._retrigger_timer = None
        self._devices = []
        self._audio_buf = np.zeros(0, dtype=np.float32) if DEPS_OK else None

        # MIDI state
        self._midi_devices = []
        self._midi_input = None        # pygame.midi.Input or rtmidi.MidiIn
        self._midi_thread = None
        self._midi_running = False
        self._rtmidi_in = None         # rtmidi.MidiIn instance kept alive while open

        # Keyboard state
        self.kb_octave = 4
        self._held_keys = {}
        self._release_timers = {}
        self._kb_bind_ids = []

        # Bank state
        self._bank_path = DEFAULT_BANK_PATH
        self._processing = False

        self.mode_var = tk.StringVar(value='keyboard')

        if DEPS_OK:
            pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(self.NUM_VOICES + 4)
            self._reverb_ir = self._build_reverb_ir()
            if MIDI_OK and not RTMIDI_OK:
                try:
                    pygame.midi.init()
                except Exception:
                    pass

        self._build_ui()

        if not DEPS_OK:
            self.root.after(200, self._show_dep_error)
        else:
            self.root.after(100, self._start)
            self.root.after(150, self._load_prefs)
            self.root.after(200, self._poll_voices)

    # ── UI ───────────────────────────────────────────────────────────────────

    def _text_canvas(self, parent, text, font, fill, bg=BG,
                     width=340, height=24, cx=None, cy=None,
                     anchor='center', wrap=0):
        """Return a Canvas with one text item. Caller must pack/grid."""
        c = tk.Canvas(parent, bg=bg, highlightthickness=0, width=width, height=height)
        x = (width // 2) if cx is None else cx
        y = (height // 2) if cy is None else cy
        kw = dict(text=text, fill=fill, font=font, anchor=anchor)
        if wrap:
            kw['width'] = wrap
        tid = c.create_text(x, y, **kw)
        return c, tid

    def _build_ui(self):
        self.root.configure(bg=BG)

        # Title
        tc, _ = self._text_canvas(self.root, "Waveform-Matched Sampler",
                                   ("Helvetica", 15, "bold"), FG, height=30)
        tc.pack(pady=(14, 4))

        # Note display
        note_c = tk.Canvas(self.root, bg=BG_DARK, width=320, height=90,
                           highlightthickness=0)
        note_c.pack(padx=16, pady=4)
        self._note_canvas = note_c
        self._note_text_id = note_c.create_text(160, 45, text="—",
                                                 fill=ACCENT,
                                                 font=("Helvetica", 44, "bold"),
                                                 anchor='center')

        # Status line
        status_c = tk.Canvas(self.root, bg=BG, width=360, height=28,
                             highlightthickness=0)
        status_c.pack(pady=(0, 4))
        self._status_canvas = status_c
        self._status_text_id = status_c.create_text(180, 14, text="Starting…",
                                                     fill=FG,
                                                     font=("Helvetica", 11),
                                                     anchor='center',
                                                     width=340)

        # Voice LEDs
        voice_row = tk.Frame(self.root, bg=BG)
        voice_row.pack(pady=4)

        vc, _ = self._text_canvas(voice_row, "Voices:", ("Helvetica", 11), FG,
                                   width=58, height=22, anchor='w', cx=2)
        vc.pack(side='left', padx=(0, 4))

        self.voice_leds = []
        for i in range(self.NUM_VOICES):
            lc = tk.Canvas(voice_row, bg=BG, width=30, height=22, highlightthickness=0)
            lc.pack(side='left', padx=2)
            rid = lc.create_rectangle(1, 1, 29, 21, fill=SURFACE, outline="")
            tid = lc.create_text(15, 11, text=str(i + 1), fill=FG_DIMR,
                                 font=("Helvetica", 11, "bold"), anchor='center')
            self.voice_leds.append({'canvas': lc, 'rect': rid, 'text': tid})

        # Now Playing
        np_frame = tk.LabelFrame(self.root, text=" Now Playing ", fg=FG,
                                 bg=BG, font=("Helvetica", 10))
        np_frame.pack(fill='x', padx=16, pady=(2, 6))
        self._voice_label_items = []
        for i in range(self.NUM_VOICES):
            row = tk.Frame(np_frame, bg=BG)
            row.pack(fill='x', padx=6, pady=1)
            num_c = tk.Canvas(row, bg=BG, width=16, height=16, highlightthickness=0)
            num_c.pack(side='left')
            num_c.create_text(8, 8, text=str(i + 1), fill=FG_DIMR,
                              font=("Helvetica", 9, "bold"), anchor='center')
            name_c = tk.Canvas(row, bg=BG, width=330, height=16, highlightthickness=0)
            name_c.pack(side='left', padx=(2, 0))
            tid = name_c.create_text(4, 8, text="—", fill=FG_DIM,
                                     font=("Helvetica", 9), anchor='w')
            self._voice_label_items.append((name_c, tid))

        # Sample Bank
        bframe = tk.LabelFrame(self.root, text=" Sample Bank ", fg=FG,
                               bg=BG, font=("Helvetica", 10))
        bframe.pack(fill='x', padx=16, pady=6)

        # Bank path row
        path_row = tk.Frame(bframe, bg=BG)
        path_row.pack(fill='x', padx=6, pady=(4, 0))
        bank_c = tk.Canvas(path_row, bg=BG, width=248, height=20, highlightthickness=0)
        bank_c.pack(side='left')
        self._bank_canvas = bank_c
        self._bank_text_id = bank_c.create_text(4, 10,
                                                  text=_short_path(DEFAULT_BANK_PATH),
                                                  fill=FG_DIM,
                                                  font=("Helvetica", 9),
                                                  anchor='w')
        tk.Button(path_row, text="Change…", command=self._change_bank,
                  bg=SURFACE, fg=FG, relief='flat',
                  activebackground=SURF2).pack(side='right', padx=4)

        # Sample count
        count_c = tk.Canvas(bframe, bg=BG, width=340, height=20, highlightthickness=0)
        count_c.pack(anchor='w', padx=8)
        self._count_canvas = count_c
        self._count_text_id = count_c.create_text(4, 10, text="",
                                                   fill=GREEN,
                                                   font=("Helvetica", 9),
                                                   anchor='w')

        # Action buttons
        btn_row = tk.Frame(bframe, bg=BG)
        btn_row.pack(fill='x', padx=6, pady=(4, 4))
        self._add_folder_btn = tk.Button(
            btn_row, text="+ Add Folder…", command=self._add_folder,
            bg=SURFACE, fg=FG, relief='flat', activebackground=SURF2)
        self._add_folder_btn.pack(side='left', padx=(0, 4))
        self._decompose_btn = tk.Button(
            btn_row, text="+ Decompose File…", command=self._decompose_file,
            bg=SURFACE, fg=FG, relief='flat', activebackground=SURF2)
        self._decompose_btn.pack(side='left', padx=4)
        self._clear_btn = tk.Button(
            btn_row, text="Clear Bank", command=self._clear_bank,
            bg=SURFACE, fg=FG_DIMR, relief='flat', activebackground=SURF2)
        self._clear_btn.pack(side='right', padx=4)

        # Progress / bank status line
        prog_c = tk.Canvas(bframe, bg=BG, width=340, height=18, highlightthickness=0)
        prog_c.pack(anchor='w', padx=8, pady=(0, 4))
        self._prog_canvas = prog_c
        self._prog_text_id = prog_c.create_text(4, 9, text="",
                                                  fill=ACCENT,
                                                  font=("Helvetica", 9),
                                                  anchor='w')

        # Input mode selector
        mframe = tk.LabelFrame(self.root, text=" Input Mode ", fg=FG,
                               bg=BG, font=("Helvetica", 10))
        mframe.pack(fill='x', padx=16, pady=6)
        mrow = tk.Frame(mframe, bg=BG)
        mrow.pack(padx=8, pady=6)
        for label, val in [("Audio", "audio"), ("MIDI", "midi"), ("Keyboard", "keyboard")]:
            tk.Radiobutton(mrow, text=label, variable=self.mode_var, value=val,
                           command=self._on_mode_change,
                           bg=BG, fg=FG, selectcolor=SURFACE,
                           activebackground=BG, activeforeground=FG,
                           font=("Helvetica", 11)).pack(side='left', padx=14)

        # Input config container — one sub-frame shown at a time
        input_container = tk.Frame(self.root, bg=BG)
        input_container.pack(fill='x', padx=16)

        # Audio sub-frame
        self.audio_frame = tk.LabelFrame(input_container, text=" Audio Input ", fg=FG,
                                         bg=BG, font=("Helvetica", 10))
        self._devices = self._get_input_devices()
        self.device_var = tk.StringVar()
        audio_combo = ttk.Combobox(self.audio_frame, textvariable=self.device_var,
                                   values=[d[1] for d in self._devices],
                                   state='readonly', width=46)
        if self._devices:
            audio_combo.current(0)
        audio_combo.pack(padx=8, pady=6)

        # MIDI sub-frame
        self.midi_frame = tk.LabelFrame(input_container, text=" MIDI Input ", fg=FG,
                                        bg=BG, font=("Helvetica", 10))
        self._midi_devices = self._get_midi_devices()
        self.midi_device_var = tk.StringVar()
        midi_row = tk.Frame(self.midi_frame, bg=BG)
        midi_row.pack(fill='x', padx=8, pady=6)
        self._midi_combo = ttk.Combobox(midi_row, textvariable=self.midi_device_var,
                                        values=[d[1] for d in self._midi_devices],
                                        state='readonly', width=36)
        if self._midi_devices:
            self._midi_combo.current(0)
        else:
            self.midi_device_var.set("No MIDI input devices found")
        self._midi_combo.pack(side='left')
        tk.Button(midi_row, text="Refresh", command=self._refresh_midi_devices,
                  bg=SURFACE, fg=FG, relief='flat',
                  activebackground=SURF2).pack(side='left', padx=(6, 0))

        # Keyboard sub-frame
        self.kb_frame = tk.LabelFrame(input_container, text=" Keyboard ", fg=FG,
                                      bg=BG, font=("Helvetica", 10))

        kb_hint_c = tk.Canvas(self.kb_frame, bg=BG, width=300, height=62,
                              highlightthickness=0)
        kb_hint_c.pack(anchor='w', padx=10, pady=(6, 2))
        kb_hint_c.create_text(4, 4, text=_KB_HINT, fill=FG_DIM,
                              font=("Courier", 10), anchor='nw')

        kb_oct_c = tk.Canvas(self.kb_frame, bg=BG, width=220, height=22,
                             highlightthickness=0)
        kb_oct_c.pack(pady=(0, 6))
        self._kb_oct_canvas = kb_oct_c
        self._kb_oct_text_id = kb_oct_c.create_text(110, 11, text="Base note: C4",
                                                      fill=ACCENT,
                                                      font=("Helvetica", 10, "bold"),
                                                      anchor='center')
        self.kb_frame.pack(fill='x')  # shown by default (keyboard mode)

        # Effects
        eframe = tk.LabelFrame(self.root, text=" Effects ", fg=FG,
                               bg=BG, font=("Helvetica", 10))
        eframe.pack(fill='x', padx=16, pady=6)
        erow = tk.Frame(eframe, bg=BG)
        erow.pack(padx=8, pady=4)
        def _fx_row(label, unit, from_, to, default, on_change):
            row = tk.Frame(eframe, bg=BG)
            row.pack(fill='x', padx=8, pady=2)
            lc = tk.Canvas(row, bg=BG, width=60, height=22, highlightthickness=0)
            lc.pack(side='left')
            lc.create_text(4, 11, text=label, fill=FG, font=("Helvetica", 10), anchor='w')
            var = tk.IntVar(value=default)
            tk.Scale(row, variable=var, from_=from_, to=to,
                     orient='horizontal', length=230, bg=BG, fg=FG,
                     troughcolor=SURFACE, highlightthickness=0,
                     command=on_change).pack(side='left')
            uc = tk.Canvas(row, bg=BG, width=30, height=22, highlightthickness=0)
            uc.pack(side='left')
            uc.create_text(4, 11, text=unit, fill=FG, font=("Helvetica", 10), anchor='w')
            return var

        self.reverb_var  = _fx_row("Reverb",   "%",  0, 100, 0,
                                   lambda v: setattr(self, 'reverb_wet', int(v) / 100.0))
        self.delay_var   = _fx_row("Delay",    "ms", 0, 1000, 0,
                                   lambda v: setattr(self, 'delay_ms', int(v)))
        self.feedback_var = _fx_row("Feedback", "%",  0,  90, 50,
                                    lambda v: setattr(self, 'delay_feedback', int(v) / 100.0))

        # Retrigger interval
        rframe = tk.LabelFrame(self.root, text=" Retrigger Interval ", fg=FG,
                               bg=BG, font=("Helvetica", 10))
        rframe.pack(fill='x', padx=16, pady=6)
        rrow = tk.Frame(rframe, bg=BG)
        rrow.pack(padx=8, pady=4)
        self.retrigger_var = tk.IntVar(value=2000)
        tk.Scale(rrow, variable=self.retrigger_var, from_=100, to=8000,
                 orient='horizontal', length=260, bg=BG, fg=FG,
                 troughcolor=SURFACE, highlightthickness=0,
                 command=lambda v: setattr(self, 'retrigger_ms', int(v))
                 ).pack(side='left')
        ms_c = tk.Canvas(rrow, bg=BG, width=28, height=22, highlightthickness=0)
        ms_c.pack(side='left')
        ms_c.create_text(4, 11, text="ms", fill=FG, font=("Helvetica", 11), anchor='w')

    # ── Canvas text update helpers ────────────────────────────────────────────

    def _set_note_text(self, text):
        self._note_canvas.itemconfig(self._note_text_id, text=text)

    def _set_status(self, text):
        self._status_canvas.itemconfig(self._status_text_id, text=text)

    def _set_progress(self, text):
        self._prog_canvas.itemconfig(self._prog_text_id, text=text)

    # ── Mode switching ────────────────────────────────────────────────────────

    def _on_mode_change(self):
        self.audio_frame.pack_forget()
        self.midi_frame.pack_forget()
        self.kb_frame.pack_forget()
        mode = self.mode_var.get()
        if mode == 'audio':
            self.audio_frame.pack(fill='x')
        elif mode == 'midi':
            self.midi_frame.pack(fill='x')
        else:
            self.kb_frame.pack(fill='x')
        self._stop()
        self._start()

    def _idle_status(self):
        mode = self.mode_var.get()
        if mode == 'audio':
            return "Listening — play into the selected input."
        elif mode == 'midi':
            return "Ready — play on your MIDI controller."
        else:
            return "Ready — use the keyboard to play notes."

    # ── Devices ──────────────────────────────────────────────────────────────

    def _get_input_devices(self):
        if not DEPS_OK:
            return []
        devices = []
        try:
            for i, d in enumerate(sd.query_devices()):
                if d['max_input_channels'] > 0:
                    devices.append((i, d['name']))
        except Exception:
            pass
        return devices

    def _get_midi_devices(self):
        if RTMIDI_OK:
            try:
                tmp = rtmidi.MidiIn()
                ports = tmp.get_ports()
                del tmp
                return list(enumerate(ports))
            except Exception:
                return []
        if not MIDI_OK:
            return []
        devices = []
        try:
            for i in range(pygame.midi.get_count()):
                info = pygame.midi.get_device_info(i)
                if info[2]:  # is_input
                    devices.append((i, info[1].decode('utf-8', errors='replace')))
        except Exception:
            pass
        return devices

    def _refresh_midi_devices(self):
        self._stop_midi()
        if RTMIDI_OK:
            # rtmidi rescans automatically on each MidiIn() instantiation
            pass
        elif MIDI_OK:
            try:
                pygame.midi.quit()
                pygame.midi.init()
            except Exception:
                pass
        self._midi_devices = self._get_midi_devices()
        names = [d[1] for d in self._midi_devices]
        self._midi_combo['values'] = names
        if self._midi_devices:
            self._midi_combo.current(0)
            self._set_status(f"Found {len(self._midi_devices)} MIDI input device(s).")
        else:
            self.midi_device_var.set("No MIDI input devices found")
            self._set_status("No MIDI input devices found.")

    # ── Sample Bank ──────────────────────────────────────────────────────────

    def _save_prefs(self, folder):
        try:
            PREFS_PATH.write_text(json.dumps({'bank_path': folder}))
        except Exception:
            pass

    def _load_prefs(self):
        folder = None
        try:
            prefs = json.loads(PREFS_PATH.read_text())
            folder = prefs.get('bank_path') or prefs.get('sample_folder')
        except Exception:
            pass
        if folder and Path(folder).is_dir():
            self._bank_path = Path(folder)
            self._load_folder(folder)
        elif DEFAULT_BANK_PATH.is_dir():
            self._bank_path = DEFAULT_BANK_PATH
            self._load_folder(str(DEFAULT_BANK_PATH))

    def _change_bank(self):
        folder = filedialog.askdirectory(title="Select Bank Location")
        if folder:
            self._bank_path = Path(folder)
            self._load_folder(folder)

    def _set_processing(self, state):
        self._processing = state
        s = 'disabled' if state else 'normal'
        self._add_folder_btn.config(state=s)
        self._decompose_btn.config(state=s)
        self._clear_btn.config(state=s)

    def _add_folder(self):
        if self._processing:
            return
        if not LIBROSA_OK:
            import tkinter.messagebox as mb
            mb.showerror("Missing librosa",
                         "Install librosa and soundfile:\n  pip install librosa soundfile")
            return
        folder = filedialog.askdirectory(title="Select Folder to Add to Bank")
        if not folder:
            return
        self._set_processing(True)
        self._set_progress("Starting…")
        threading.Thread(target=self._add_folder_worker, args=(folder,), daemon=True).start()

    def _add_folder_worker(self, folder):
        import numpy as _np
        AUDIO_EXT = {'.wav', '.aif', '.aiff', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
        NEEDS_CONV = {'.mp3', '.flac', '.ogg', '.m4a', '.aac'}

        src = Path(folder)
        files = sorted(f for f in src.rglob('*')
                       if f.is_file() and f.suffix.lower() in AUDIO_EXT)

        if not files:
            self.root.after(0, self._set_progress, "No audio files found.")
            self.root.after(0, self._set_processing, False)
            return

        self._bank_path.mkdir(parents=True, exist_ok=True)
        done = 0

        for i, fp in enumerate(files, 1):
            self.root.after(0, self._set_progress,
                            f"Detecting pitch: {fp.name} ({i}/{len(files)})…")
            midi = self._detect_pitch_lib(fp)
            if midi is None:
                folder_out = self._bank_path / '_undetected'
            else:
                folder_out = self._bank_path / midi_to_name(midi)
            folder_out.mkdir(parents=True, exist_ok=True)

            out_suffix = '.wav' if fp.suffix.lower() in NEEDS_CONV else fp.suffix
            dest = folder_out / (fp.stem + out_suffix)
            n = 1
            while dest.exists():
                dest = folder_out / f"{fp.stem}_{n}{out_suffix}"
                n += 1

            if fp.suffix.lower() in NEEDS_CONV:
                y, sr = librosa.load(str(fp), sr=None, mono=False)
                if y.ndim > 1:
                    y = y.T
                sf.write(str(dest), y, sr)
            else:
                shutil.copy2(fp, dest)
            done += 1

        self.root.after(0, self._set_progress, f"Added {done} file(s) to bank.")
        self.root.after(0, self._load_folder, str(self._bank_path))
        self.root.after(0, self._set_processing, False)

    def _decompose_file(self):
        if self._processing:
            return
        if not LIBROSA_OK:
            import tkinter.messagebox as mb
            mb.showerror("Missing librosa",
                         "Install librosa and soundfile:\n  pip install librosa soundfile")
            return
        filetypes = [
            ("Audio files", "*.wav *.aif *.aiff *.mp3 *.flac *.ogg *.m4a *.aac"),
            ("All files", "*.*"),
        ]
        filepath = filedialog.askopenfilename(
            title="Select Audio File to Decompose", filetypes=filetypes)
        if not filepath:
            return
        self._set_processing(True)
        self._set_progress("Starting…")
        threading.Thread(target=self._decompose_worker, args=(filepath,), daemon=True).start()

    def _decompose_worker(self, filepath):
        import warnings
        import numpy as _np
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)

        src = Path(filepath)
        self.root.after(0, self._set_progress, f"Loading {src.name}…")

        try:
            y, sr = librosa.load(str(src), sr=None, mono=True)
        except Exception as e:
            self.root.after(0, self._set_progress, f"Error loading: {e}")
            self.root.after(0, self._set_processing, False)
            return

        self.root.after(0, self._set_progress, "Detecting onsets…")
        slices = self._find_slices(y, sr)

        if not slices:
            self.root.after(0, self._set_progress, "No slices detected.")
            self.root.after(0, self._set_processing, False)
            return

        self._bank_path.mkdir(parents=True, exist_ok=True)
        stem = src.stem
        done = 0

        for i, (start, end) in enumerate(slices, 1):
            seg = y[start:end]
            self.root.after(0, self._set_progress,
                            f"Processing slice {i}/{len(slices)}…")
            midi = self._detect_pitch_seg(seg, sr)
            if midi is None:
                folder_out = self._bank_path / '_undetected'
            else:
                folder_out = self._bank_path / midi_to_name(midi)
            folder_out.mkdir(parents=True, exist_ok=True)

            filename = f"{stem}_{i:03d}.wav"
            dest = folder_out / filename
            n = 1
            while dest.exists():
                dest = folder_out / f"{stem}_{i:03d}_{n}.wav"
                n += 1

            # Load stereo slice for output
            y_slice, _ = librosa.load(str(src), sr=sr, mono=False,
                                       offset=start / sr,
                                       duration=(end - start) / sr)
            if y_slice.ndim == 1:
                sf.write(str(dest), self._fade_edges(y_slice, sr), sr)
            else:
                faded = _np.stack([self._fade_edges(y_slice[ch], sr)
                                   for ch in range(y_slice.shape[0])], axis=1)
                sf.write(str(dest), faded, sr)
            done += 1

        self.root.after(0, self._set_progress,
                        f"Added {done} slice(s) from {src.name}.")
        self.root.after(0, self._load_folder, str(self._bank_path))
        self.root.after(0, self._set_processing, False)

    def _clear_bank(self):
        import tkinter.messagebox as mb
        if not mb.askyesno("Clear Bank",
                           "Delete all samples from the bank?\nThis cannot be undone."):
            return
        try:
            if self._bank_path.is_dir():
                shutil.rmtree(self._bank_path)
            self._bank_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self._set_status(f"Clear failed: {e}")
            return
        self.sample_db = {}
        self._sound_names = {}
        self._fx_cache = {}
        self._count_canvas.itemconfig(self._count_text_id, text="Bank cleared.")
        self._set_status("Sample bank cleared.")

    # ── Pitch / onset helpers (librosa-based) ─────────────────────────────────

    def _detect_pitch_lib(self, filepath):
        """Pitch-detect a file with librosa. Returns MIDI int or None."""
        try:
            y, sr = librosa.load(str(filepath), duration=4.0, mono=True)
            return self._detect_pitch_seg(y, sr)
        except Exception:
            return None

    def _detect_pitch_seg(self, y, sr):
        """Pitch-detect a numpy audio array. Returns MIDI int or None."""
        import numpy as _np
        try:
            y_harm = librosa.effects.harmonic(y)
            pitches, mags = librosa.piptrack(y=y_harm, sr=sr, threshold=0.1)
            hits = []
            for t in range(pitches.shape[1]):
                idx = mags[:, t].argmax()
                freq = pitches[idx, t]
                mag = mags[idx, t]
                if 20.0 < freq < 8000.0:
                    hits.append((freq, mag))
            if not hits:
                return None
            hits.sort(key=lambda x: x[1], reverse=True)
            top = [h[0] for h in hits[:max(1, len(hits) // 3)]]
            midi = round(69 + 12 * _np.log2(_np.median(top) / 440.0))
            return int(midi) if 0 <= midi <= 127 else None
        except Exception:
            return None

    def _find_slices(self, y, sr, min_dur=0.5, max_dur=2.0):
        """Return list of (start_sample, end_sample) tuples via onset detection."""
        import numpy as _np
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, backtrack=True,
            pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.07, wait=10
        )
        onset_samples = librosa.frames_to_samples(onset_frames).tolist()
        boundaries = sorted(set([0] + onset_samples + [len(y)]))
        min_samp = int(min_dur * sr)
        max_samp = int(max_dur * sr)
        raw_slices = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            if end - start < min_samp:
                continue
            length = end - start
            if length > max_samp:
                n_chunks = int(_np.ceil(length / max_samp))
                chunk_len = length // n_chunks
                for c in range(n_chunks):
                    cs = start + c * chunk_len
                    ce = cs + chunk_len if c < n_chunks - 1 else end
                    if ce - cs >= min_samp:
                        raw_slices.append((cs, ce))
            else:
                raw_slices.append((start, end))
        return raw_slices

    def _fade_edges(self, audio, sr, fade_in=0.005, fade_out=0.03):
        import numpy as _np
        n_in  = min(int(sr * fade_in),  len(audio) // 4)
        n_out = min(int(sr * fade_out), len(audio) // 4)
        result = audio.copy()
        if n_in > 0:
            result[:n_in]  *= _np.linspace(0.0, 1.0, n_in)
        if n_out > 0:
            result[-n_out:] *= _np.linspace(1.0, 0.0, n_out)
        return result

    # ── Sample loading ────────────────────────────────────────────────────────

    def _load_folder(self, folder):
        self.sample_db = {}
        self._sound_names = {}
        self._fx_cache = {}
        path = Path(folder)
        exts = {'.wav', '.aif', '.aiff'}
        errors = []

        # Layout 1: subfolders named by note
        for sub in path.iterdir():
            if sub.is_dir():
                midi = name_to_midi(sub.name)
                if midi is not None:
                    files = [f for f in sub.iterdir() if f.suffix.lower() in exts]
                    if files:
                        sounds, errs = self._load_sounds(files)
                        errors.extend(errs)
                        if sounds:
                            self.sample_db.setdefault(midi, []).extend(sounds)

        # Layout 2: flat files, stem encodes note
        for f in path.iterdir():
            if f.is_file() and f.suffix.lower() in exts:
                stem = f.stem.split('_')[0]
                midi = name_to_midi(stem)
                if midi is not None:
                    sounds, errs = self._load_sounds([f])
                    errors.extend(errs)
                    if sounds:
                        self.sample_db.setdefault(midi, []).extend(sounds)

        total = sum(len(v) for v in self.sample_db.values())
        notes = len(self.sample_db)
        self._bank_canvas.itemconfig(self._bank_text_id,
                                     text=_short_path(folder))
        self._count_canvas.itemconfig(self._count_text_id,
                                      text=f"{total} samples across {notes} notes loaded")
        if total == 0 and errors:
            self._set_status(f"Failed to load samples: {errors[0]}")
        elif total == 0:
            self._set_status("No recognisable samples found in bank.")
        else:
            self._set_status(f"Loaded {total} samples for {notes} MIDI notes.")
            self._save_prefs(folder)

    def _load_sounds(self, files):
        sounds, errors = [], []
        for f in files:
            try:
                s = pygame.mixer.Sound(str(f))
                self._sound_names[id(s)] = f.name
                sounds.append(s)
            except Exception as e:
                errors.append(f"{f.name}: {e}")
        return sounds, errors

    # ── Start / Stop ─────────────────────────────────────────────────────────

    def _start(self):
        mode = self.mode_var.get()
        if mode == 'audio':
            self._start_audio()
        elif mode == 'midi':
            self._start_midi()
        else:
            self._start_keyboard()

    def _stop(self):
        self._cancel_retrigger()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self._stop_midi()
        self._stop_keyboard()
        self.running = False
        self.current_note = -1
        self._set_note_text("—")
        self._reset_leds()

    # ── Audio mode ───────────────────────────────────────────────────────────

    def _start_audio(self):
        device_idx = None
        selected = self.device_var.get()
        for idx, name in self._devices:
            if name == selected:
                device_idx = idx
                break
        try:
            self._audio_buf = np.zeros(0, dtype=np.float32)
            self.stream = sd.InputStream(
                device=device_idx,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=HOP_SIZE,
                dtype='float32',
                callback=self._audio_callback,
            )
            self.stream.start()
            self.running = True
            self._set_status(self._idle_status())
        except Exception as e:
            self._set_status(f"Error opening audio: {e}")

    def _audio_callback(self, indata, frames, time_info, status):
        samples = indata[:, 0].astype(np.float32)
        self._audio_buf = np.concatenate([self._audio_buf, samples])
        if len(self._audio_buf) < BUFFER_SIZE:
            return
        frame = self._audio_buf[:BUFFER_SIZE]
        self._audio_buf = self._audio_buf[HOP_SIZE:]

        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms < 0.005:
            self.root.after(0, self._on_silence)
            return

        freq, conf = _yin_pitch(frame, SAMPLE_RATE)

        if freq > 20 and conf > 0.7:
            midi = int(round(69 + 12 * np.log2(freq / 440.0)))
            if 0 <= midi <= 127:
                self.root.after(0, self._on_note, midi)
                return
        self.root.after(0, self._on_silence)

    # ── MIDI mode ────────────────────────────────────────────────────────────

    def _start_midi(self):
        if not self._midi_devices:
            self._set_status("No MIDI input devices found. Click Refresh.")
            return
        device_idx = None
        selected = self.midi_device_var.get()
        for idx, name in self._midi_devices:
            if name == selected:
                device_idx = idx
                break
        if device_idx is None:
            self._set_status("Select a MIDI input device first.")
            return
        try:
            if RTMIDI_OK:
                self._rtmidi_in = rtmidi.MidiIn()
                self._rtmidi_in.open_port(device_idx)
                self._rtmidi_in.ignore_types(sysex=True, timing=True, active_sense=True)
            else:
                self._midi_input = pygame.midi.Input(device_idx)
            self._midi_running = True
            self._midi_thread = threading.Thread(target=self._midi_poll_loop, daemon=True)
            self._midi_thread.start()
            self.running = True
            self._set_status(self._idle_status())
        except Exception as e:
            self._set_status(f"Error opening MIDI device: {e}")

    def _midi_poll_loop(self):
        while self._midi_running:
            try:
                if RTMIDI_OK:
                    msg = self._rtmidi_in.get_message() if self._rtmidi_in else None
                    if msg:
                        data, _ = msg
                        status = data[0] & 0xF0
                        note   = data[1] if len(data) > 1 else 0
                        vel    = data[2] if len(data) > 2 else 0
                        if status == 0x90 and vel > 0:
                            self.root.after(0, self._on_note, note)
                        elif status == 0x80 or (status == 0x90 and vel == 0):
                            self.root.after(0, self._on_midi_note_off, note)
                else:
                    if self._midi_input and self._midi_input.poll():
                        for event in self._midi_input.read(16):
                            status = event[0][0] & 0xF0
                            note   = event[0][1]
                            vel    = event[0][2]
                            if status == 0x90 and vel > 0:
                                self.root.after(0, self._on_note, note)
                            elif status == 0x80 or (status == 0x90 and vel == 0):
                                self.root.after(0, self._on_midi_note_off, note)
            except Exception:
                break
            time.sleep(0.001)

    def _on_midi_note_off(self, note):
        if note == self.current_note:
            self._on_silence()

    def _stop_midi(self):
        self._midi_running = False
        if self._rtmidi_in:
            try:
                self._rtmidi_in.close_port()
            except Exception:
                pass
            self._rtmidi_in = None
        if self._midi_input:
            try:
                self._midi_input.close()
            except Exception:
                pass
            self._midi_input = None

    # ── Keyboard mode ────────────────────────────────────────────────────────

    def _start_keyboard(self):
        self._held_keys = {}
        self.root.bind_all('<KeyPress>', self._on_key_press)
        self.root.bind_all('<KeyRelease>', self._on_key_release)
        self.running = True
        self._set_status(self._idle_status())

    def _stop_keyboard(self):
        try:
            self.root.unbind_all('<KeyPress>')
            self.root.unbind_all('<KeyRelease>')
        except Exception:
            pass
        self._kb_bind_ids = []
        self._held_keys = {}
        for tid in self._release_timers.values():
            self.root.after_cancel(tid)
        self._release_timers = {}

    def _on_key_press(self, event):
        key = event.keysym.lower()
        # Cancel any pending release debounce — key-repeat on macOS fires
        # KeyRelease then KeyPress in quick succession rather than held KeyPress.
        if key in self._release_timers:
            self.root.after_cancel(self._release_timers.pop(key))
        if key == 'z':
            self.kb_octave = max(0, self.kb_octave - 1)
            self._update_kb_display()
            return
        if key == 'x':
            self.kb_octave = min(8, self.kb_octave + 1)
            self._update_kb_display()
            return
        if key in _KB_OFFSETS and key not in self._held_keys:
            midi = (self.kb_octave + 1) * 12 + _KB_OFFSETS[key]
            if 0 <= midi <= 127:
                self._held_keys[key] = midi
                self._on_note(midi)

    def _on_key_release(self, event):
        key = event.keysym.lower()
        if key in self._held_keys:
            tid = self.root.after(50, self._do_key_release, key)
            self._release_timers[key] = tid

    def _do_key_release(self, key):
        self._release_timers.pop(key, None)
        note = self._held_keys.pop(key, None)
        if note is not None and note == self.current_note:
            self._on_silence()

    def _update_kb_display(self):
        base_midi = (self.kb_octave + 1) * 12
        self._kb_oct_canvas.itemconfig(self._kb_oct_text_id,
                                        text=f"Base note: {midi_to_name(base_midi)}")

    # ── Note events ──────────────────────────────────────────────────────────

    def _on_note(self, midi):
        if midi != self.current_note:
            self.current_note = midi
            name = midi_to_name(midi)
            self._set_note_text(f"{name}  ({midi})")
            self._set_status(f"Playing {name} ({midi})")
            self._cancel_retrigger()
            self._trigger(midi)
            if self.mode_var.get() == 'audio':
                self._schedule_retrigger(midi)

    def _on_silence(self):
        if self.current_note != -1:
            self.current_note = -1
            self._set_note_text("—")
            self._set_status(self._idle_status())
            self._cancel_retrigger()

    # ── Playback ─────────────────────────────────────────────────────────────

    def _trigger(self, midi):
        sound = self._pick_sample(midi)
        if sound is None:
            name = midi_to_name(midi)
            if not self.sample_db:
                self._set_status(f"{name} ({midi}) — no samples loaded.")
            else:
                self._set_status(f"{name} ({midi}) — no matching sample found.")
            return
        if self.reverb_wet > 0 or self.delay_ms > 0:
            sound = self._get_processed_sound(sound)
        voice = self.voice_idx % self.NUM_VOICES
        self.voice_idx += 1
        channel = pygame.mixer.Channel(voice)
        channel.play(sound)
        name = self._sound_names.get(id(sound), '?')
        self._voice_samples[voice] = name
        self._set_voice_label(voice, name, active=True)
        self._flash_led(voice)

    def _pick_sample(self, midi):
        if midi in self.sample_db:
            return random.choice(self.sample_db[midi])
        best, best_dist = None, 999
        for note, sounds in self.sample_db.items():
            d = abs(note - midi)
            if d < best_dist and sounds:
                best_dist, best = d, sounds
        if best and best_dist <= 12:
            return random.choice(best)
        pitch_class = midi % 12
        same_class = [s for note, sounds in self.sample_db.items()
                      if note % 12 == pitch_class for s in sounds]
        return random.choice(same_class) if same_class else None

    def _schedule_retrigger(self, midi):
        def fire():
            if self.current_note == midi:
                self.root.after(0, self._trigger, midi)
                self._schedule_retrigger(midi)
        self._retrigger_timer = threading.Timer(self.retrigger_ms / 1000.0, fire)
        self._retrigger_timer.daemon = True
        self._retrigger_timer.start()

    def _cancel_retrigger(self):
        if self._retrigger_timer:
            self._retrigger_timer.cancel()
            self._retrigger_timer = None

    # ── Reverb ────────────────────────────────────────────────────────────────

    def _build_reverb_ir(self):
        n = int(SAMPLE_RATE * 1.4)
        t = np.linspace(0, 1, n)
        rng = np.random.default_rng(42)
        ir = rng.standard_normal(n).astype(np.float32)
        ir *= np.exp(-4.5 * t)
        ir[0] = 1.0
        ir /= np.max(np.abs(ir))
        return ir

    def _get_processed_sound(self, sound):
        wet      = round(self.reverb_wet     * 20) / 20   # 5% steps
        delay    = round(self.delay_ms       / 10) * 10   # 10ms steps
        feedback = round(self.delay_feedback * 20) / 20   # 5% steps
        cache_key = (id(sound), wet, delay, feedback)
        if cache_key in self._fx_cache:
            return self._fx_cache[cache_key]
        try:
            data = pygame.sndarray.array(sound)
            if delay > 0:
                data = self._apply_delay(data, delay, feedback)
            if wet > 0:
                data = self._apply_reverb(data, wet)
            new_sound = pygame.sndarray.make_sound(data)
            self._sound_names[id(new_sound)] = self._sound_names.get(id(sound), '?')
            self._fx_cache[cache_key] = new_sound
            return new_sound
        except Exception:
            return sound

    def _apply_delay(self, data, delay_ms, feedback):
        """Add feedback echoes to int16 array."""
        delay_samples = int(SAMPLE_RATE * delay_ms / 1000.0)
        if delay_samples == 0 or feedback <= 0:
            return data
        data_f = data.astype(np.float32) / 32768.0
        stereo = data_f.ndim == 2
        channels = [data_f[:, ch] for ch in range(data_f.shape[1])] if stereo else [data_f]
        n = len(channels[0])
        n_taps = max(1, int(np.log(0.001) / np.log(max(feedback, 1e-8))))
        n_out = n + n_taps * delay_samples
        out = []
        for ch in channels:
            result = np.zeros(n_out, dtype=np.float32)
            result[:n] = ch
            for tap in range(1, n_taps + 1):
                start = tap * delay_samples
                result[start:start + n] += ch * (feedback ** tap)
            out.append(result)
        result_arr = np.stack(out, axis=1) if stereo else out[0]
        return np.clip(result_arr * 32768.0, -32768, 32767).astype(np.int16)

    def _apply_reverb(self, data, wet):
        """Convolve int16 stereo/mono array with IR; return int16 array of same shape."""
        data_f = data.astype(np.float32) / 32768.0
        stereo = data_f.ndim == 2
        channels = [data_f[:, ch] for ch in range(data_f.shape[1])] if stereo else [data_f]

        n = len(channels[0])
        ir = self._reverb_ir
        n_out = n + len(ir) - 1
        fft_len = 1 << (n_out - 1).bit_length()

        if fft_len not in self._reverb_ir_ffts:
            self._reverb_ir_ffts[fft_len] = np.fft.rfft(ir, n=fft_len)
        ir_fft = self._reverb_ir_ffts[fft_len]

        out = []
        for ch in channels:
            wet_sig = np.fft.irfft(np.fft.rfft(ch, n=fft_len) * ir_fft)[:n_out]
            dry_rms = np.sqrt(np.mean(ch ** 2)) + 1e-8
            wet_rms = np.sqrt(np.mean(wet_sig ** 2)) + 1e-8
            wet_sig *= dry_rms / wet_rms
            dry_pad = np.zeros(n_out, dtype=np.float32)
            dry_pad[:n] = ch
            out.append((1 - wet) * dry_pad + wet * wet_sig)

        result = np.stack(out, axis=1) if stereo else out[0]
        return np.clip(result * 32768.0, -32768, 32767).astype(np.int16)

    # ── Now-playing display ───────────────────────────────────────────────────

    def _set_voice_label(self, idx, name, active=False):
        canvas, tid = self._voice_label_items[idx]
        display = name if len(name) <= 46 else name[:43] + '…'
        canvas.itemconfig(tid, text=display, fill=FG if active else FG_DIM)

    def _poll_voices(self):
        for i in range(self.NUM_VOICES):
            if self._voice_samples[i] != '—' and not pygame.mixer.Channel(i).get_busy():
                self._voice_samples[i] = '—'
                self._set_voice_label(i, '—', active=False)
        self.root.after(100, self._poll_voices)

    # ── LED helpers ──────────────────────────────────────────────────────────

    def _flash_led(self, idx):
        led = self.voice_leds[idx]
        led['canvas'].itemconfig(led['rect'], fill=ACCENT)
        led['canvas'].itemconfig(led['text'], fill=BG_DARK)
        self.root.after(250, lambda: (
            led['canvas'].itemconfig(led['rect'], fill=SURFACE),
            led['canvas'].itemconfig(led['text'], fill=FG_DIMR)
        ))

    def _reset_leds(self):
        for led in self.voice_leds:
            led['canvas'].itemconfig(led['rect'], fill=SURFACE)
            led['canvas'].itemconfig(led['text'], fill=FG_DIMR)

    # ── Misc ─────────────────────────────────────────────────────────────────

    def _show_dep_error(self):
        import tkinter.messagebox as mb
        mb.showerror(
            "Missing Dependencies",
            f"Could not import: {MISSING}\n\n"
            "Install everything with:\n\n"
            "  pip install sounddevice pygame numpy\n\n"
            "Then relaunch the app."
        )

    def on_close(self):
        self._stop()
        if DEPS_OK:
            pygame.mixer.quit()
            if MIDI_OK and not RTMIDI_OK:
                try:
                    pygame.midi.quit()
                except Exception:
                    pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SamplerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
