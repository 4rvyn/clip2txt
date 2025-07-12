#!/usr/bin/env python3
"""
link/local clip + timestamps ► MP3/WAV pipeline ► local Whisper transcript
===========================================

REQUIREMENTS (tested on Windows)
------------------------------------------
1. Python ≥ 3.11
2. ffmpeg.exe on PATH – https://www.gyan.dev/ffmpeg/builds/
3. Inside your venv run:

   pip install -r requirements.txt
   # ONYL GPU users: install the matching CUDA wheel first, e.g.
   # pip install torch --index-url https://download.pytorch.org/whl/cu121

USER SETTINGS
-------------
Fill in the variables below.  Leave OUTDIR empty ("") to default
to the script’s own directory.
"""

import pathlib
import re
import sys
import time
from datetime import datetime
import subprocess
from faster_whisper import WhisperModel
from faster_whisper import BatchedInferencePipeline
import os
import yt_dlp
import numpy as np
import multiprocessing as mp

threads = str(mp.cpu_count())
os.environ["OMP_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads

# ────────── USER SETTINGS ────────── #
SOURCES  = [
    r"https://www.youtube.com/watch?v=CPBJgpK0Ulc", # any URL or path to local file
    # r"C:\path\to\local\video.mp4",  # local video
    # r"C:\path\to\local\audio.mp3",  # local audio file
]


START_TS = ""  # HH:MM:SS (empty → 00:00:00)
END_TS   = ""  # HH:MM:SS (empty → clip end)
OUTDIR   = r"output"   # "" → script folder
MODEL = 'medium'  # use smaller Whisper model -> faster, less accurate) + ".en" for English etc.
# Available models: tiny, base, small, medium, large-v1, large-v2
COMP_TYPE = 'int8_float32'  # 'int8', 'float16', 'int16', 'float32', 'int8_float32' ...

TS_RE = re.compile(r"^(\d\d):([0-5]\d):([0-5]\d)$")

# ───────────────── helpers ────────────────── #
def ts_to_sec(ts: str) -> int:
    m = TS_RE.match(ts)
    if not m:
        raise ValueError(f"Bad timestamp {ts!r}. Use HH:MM:SS.")
    h, m_, s = map(int, m.groups())
    return h * 3600 + m_ * 60 + s

def sec_to_ts(sec: int) -> str:
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def ensure_outdir(path_hint: str) -> pathlib.Path:
    d = pathlib.Path(path_hint) if path_hint else pathlib.Path(__file__).parent
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve()

def get_video_duration(url: str) -> int | None:
    """Ask yt‑dlp (quietly) for duration in seconds."""
    opts = {'quiet': True, 'get_duration': True, 'nocheckcertificate': True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            return int(ydl.extract_info(url, download=False).get('duration') or 0) or None
    except Exception:
        return None

def extract_audio_local(src: pathlib.Path, start: str, end: str, dest_mp3: pathlib.Path):
    """ffmpeg-trim any local mp3/mp4 to an MP3 clip."""
    print('[1/2] Extracting local clip… Please wait…')
    cmd = [
        'ffmpeg', '-v', 'error',  # keep ffmpeg quiet
        '-ss', start, *(['-to', end] if end else []),
        '-i', str(src),
        '-vn', '-acodec', 'libmp3lame',  # audio-only → mp3
        str(dest_mp3)
    ]
    run = subprocess.run(cmd)
    if run.returncode != 0 or not dest_mp3.exists():
        raise RuntimeError('ffmpeg failed – see messages above.')
    print(f'✔ Clip saved to {dest_mp3}')

def load_audio_from_mp3(mp3_path: pathlib.Path) -> np.ndarray:
    """
    Runs ffmpeg to decode mp3_path into a float32 PCM array
    (16 kHz, mono). Returns a 1-D numpy array in [-1.0, +1.0].
    """
    cmd = [
        'ffmpeg', '-v', 'error',
        '-i', str(mp3_path),       # input
        '-ac', '1',                # mono
        '-ar', '16000',            # 16 kHz
        '-f', 'f32le',             # raw float32 little-endian
        'pipe:1'                   # → stdout
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    raw, _ = proc.communicate()
    # Interpret as float32
    audio = np.frombuffer(raw, dtype=np.float32)
    return audio


def get_local_duration(path: pathlib.Path) -> int | None:
    try:
        out = subprocess.check_output(
            ['ffprobe', '-v', 'error', '-show_entries',
             'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
             str(path)], text=True)
        return int(float(out.strip()))
    except Exception:
        return None



# ───────── yt‑dlp download ───────── #

def download_audio(url: str, start: str, end: str, dest_mp3: pathlib.Path):
    print('[1/2] Downloading & extracting clip…')

    cmd = [
        sys.executable, '-m', 'yt_dlp',
        url,
        '-S', 'hasaud,+filesize',
        '-x', '--audio-format', 'mp3',
        '--download-sections', f"*{start}-{end}",
        '-o', 'audio.%(ext)s',
        '-P', str(dest_mp3.parent)
    ]

    # inherit stdout/stderr → full logging visible (incl. keyboard buffer)
    run = subprocess.run(cmd)
    if run.returncode != 0:
        raise RuntimeError('yt‑dlp failed – see messages above.')

    if not dest_mp3.exists():
        raise RuntimeError('yt‑dlp finished but audio.mp3 not found.')

    print(f'✔ Clip saved to {dest_mp3}')


# ───────── Whisper transcription ───────── #

def write_transcript(segments: list, file_path: pathlib.Path, source: str, start: str, end: str):
    """Writes the transcript segments to a text file."""
    # You could also format this as an SRT file, JSON, etc.
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f'Transcript of: "{source}"\n')
        f.write(f"Full duration: [{start} --> {end}]\n\n")
        for seg in segments:
            # Simple text output
            start_ts = sec_to_ts(int(seg['start']))
            end_ts = sec_to_ts(int(seg['end']))
            f.write(f"[{start_ts} --> {end_ts}] {seg['text'].strip()}\n")

def transcribe(mp3_path: pathlib.Path,
               txt_path: pathlib.Path,
               model_name: str,
               compute_type: str,
               source: str,
               start: str,
               end: str):
    """Transcribes the audio using faster-whisper and handles interrupts gracefully."""
    print('\n[2/2] Transcribing audio …')
    t0 = time.time()

    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    except ImportError:
        device = 'cpu'

    print(f'▶ Loading Whisper "{model_name}" on {device.upper()} with {compute_type} …')
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        # Set to a number that works for your CPU, os.cpu_count() is a good default
        cpu_threads=os.cpu_count() or 4
    )

    pipe  = BatchedInferencePipeline(model=model)
    completed_segments = []
    status = 'Unknown'
    audio = load_audio_from_mp3(mp3_path)

    try:
        # We set log_progress=True to see the progress bar in the console.
        segments, _ = pipe.transcribe(
            audio,
            beam_size=3,        # greedy, 1 = fastest, 5 default
            patience=1.2,       # 1.0 = default, 1.2 = more accurate
            vad_filter=True,    # skip silence
            log_progress=True,
            multilingual=True,
            batch_size=8,       # decode x# segments in parallel (my cpu has 8 cores)
        )

        print("▶ Press Ctrl+C to interrupt and save partial progress.")
        
        # The magic happens here: we process the generator and store results as they come.
        for segment in segments:
            # Print to screen in real-time
            print(f"\n[{sec_to_ts(int(segment.start))} --> {sec_to_ts(int(segment.end))}] {segment.text}")

            # Store the structured data
            completed_segments.append({
                'text': segment.text,
                'start': segment.start,
                'end': segment.end
            })

        print('\n▶ Transcription complete – writing file…')
        write_transcript(completed_segments, txt_path, source, start, end)
        status = 'Completed'

    except KeyboardInterrupt:
        print('\n▶ Keyboard interrupt – saving partial transcript…')
        # If interrupted, write whatever we have collected so far.
        if completed_segments:
            write_transcript(completed_segments, txt_path, source, start, end)
            status = 'Interrupted (Partial)'
        else:
            print("  No segments were transcribed before interruption.")
            status = 'Interrupted (No Data)'
        raise    # let main() know we bailed out

    finally:
        dt = time.time() - t0
        print(f"\n✔ Transcript ({status}) saved to: {txt_path}")
        if dt > 0:
            print(f"  Took {dt/60:.1f} min ({dt:.0f} s).")


# ──────────── main ──────────── #
def main():
    print('--- SCRIPT START ---')
    try:
        out_root = ensure_outdir(OUTDIR)
        run_dir  = out_root / f"run_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir.mkdir()
        print(f'▶ Output → {run_dir}')

        mp3_path = run_dir / 'audio.mp3'
        txt_path = run_dir / 'transcript.txt'

        src_path = pathlib.Path(SOURCE).expanduser()
        is_local = src_path.is_file()

        # — determine duration & clip range —
        if is_local:
            dur = get_local_duration(src_path)
        else:
            dur = get_video_duration(SOURCE)

        if dur:
            s_sec = ts_to_sec(START_TS) if START_TS else 0
            e_sec = ts_to_sec(END_TS) if END_TS else dur
            if e_sec > dur:
                e_sec = dur
            if s_sec >= e_sec:
                s_sec, e_sec = 0, dur
            start, end = sec_to_ts(s_sec), sec_to_ts(e_sec)
        else:
            if not START_TS or not END_TS:
                raise ValueError('Unknown media length – specify both START_TS and END_TS.')
            start, end = START_TS, END_TS

        print(f'▶ Clip range: {start} → {end}')

        # — fetch or extract audio —
        if is_local:
            extract_audio_local(src_path.resolve(), start, end, mp3_path)
        else:
            download_audio(SOURCE, start, end, mp3_path)

        # — Whisper transcription —

        transcribe(mp3_path, txt_path, MODEL, COMP_TYPE, SOURCE, start, end)
        print('\n--- SCRIPT FINISHED SUCCESSFULLY ---')

    except KeyboardInterrupt:
        print('\n--- SCRIPT END (INTERRUPTED BY USER) ---')
        sys.exit(0)
    except Exception as e:
        print('\n--- SCRIPT FAILED ---')
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    for SOURCE in SOURCES:
        main()
