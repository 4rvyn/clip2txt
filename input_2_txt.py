#!/usr/bin/env python3
"""
link/local clip ► MP3 ► local Whisper transcript
===========================================

REQUIREMENTS (tested on Windows)
------------------------------------------
1. Python ≥ 3.11
2. ffmpeg.exe on PATH – https://www.gyan.dev/ffmpeg/builds/
3. Inside your venv run:

   # stable pieces
   pip install -U yt-dlp

   # Whisper – PyPI tarball is broken, grab the wheel from GitHub
   pip install git+https://github.com/openai/whisper.git@main

   # GPU users add the matching CUDA wheel first, e.g.
   # pip install torch --index-url https://download.pytorch.org/whl/cu121

USER SETTINGS
-------------
Fill in the four variables below.  Leave OUTDIR empty ("") to default
to the script’s own directory.
"""

import contextlib
import io
import pathlib
import re
import sys
import time
from datetime import datetime

import whisper
import yt_dlp

# ────────── USER SETTINGS ────────── #
URL        = "https://www.youtube.com/watch?v=L45Q1_psDqk"
START_TS   = "00:22:30"          # HH:MM:SS
END_TS     = "02:24:50"          # HH:MM:SS
OUTDIR     = r"output"
# ─────────────────────────────────── #

TS_RE = re.compile(r"^(\d\d):([0-5]\d):([0-5]\d)$")
# Regex to capture text from Whisper's verbose output for partial saves
WHISPER_VERBOSE_RE = re.compile(
    r"\[\d{2}:\d{2}(?::\d{2})?\.\d{3}\s*-->\s*\d{2}:\d{2}(?::\d{2})?\.\d{3}\]\s+(.*)"
)


# ───────── helper / validation ────── #

class Tee:
    """A helper class to duplicate a stream to multiple destinations (like a file and stdout)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class YtdlpLogger:
    """A custom yt-dlp logger to replicate the script's original output behavior."""
    def __init__(self):
        self.extracting_announced = False

    def _log(self, msg, file):
        # Filter out download percentage updates
        if msg.lstrip().startswith('[download]') and '%' in msg:
            return

        # Custom message for duration estimation
        if 'Estimating duration from bitrate, this may be inaccurate' in msg:
            print("[info] Please wait whilst it's estimating duration from bitrate, this may be inaccurate")
            file.write(msg + '\n')
            return

        # Custom message for audio extraction
        if msg.startswith('[ExtractAudio]') and not self.extracting_announced:
            print("[info] Please wait whilst extracting...")
            self.extracting_announced = True

        file.write(msg + '\n')
        file.flush()

    def debug(self, msg):
        # Skip yt-dlp's internal debugging messages
        if msg.startswith('[debug] '):
            return
        # The original script redirected stderr to stdout, so we send all output there.
        self._log(msg, sys.stdout)

    def info(self, msg):
        self._log(msg, sys.stdout)

    def warning(self, msg):
        self._log(msg, sys.stdout)

    def error(self, msg):
        # Errors from yt-dlp are also sent to stdout to match original behavior.
        self._log(msg, sys.stdout)


def ts_to_sec(ts: str) -> int:
    """Converts HH:MM:SS timestamp to total seconds."""
    m = TS_RE.match(ts)
    if not m:
        raise ValueError(f"Bad timestamp {ts!r}. Use HH:MM:SS format.")
    h, m_, s = map(int, m.groups())
    return h * 3600 + m_ * 60 + s


def ensure_outdir(path_hint: str) -> pathlib.Path:
    """Ensures the output directory exists, creating it if necessary."""
    out = pathlib.Path(path_hint) if path_hint else pathlib.Path(__file__).parent
    out.mkdir(parents=True, exist_ok=True)
    return out.resolve()


def get_video_duration(url: str) -> int | None:
    """Fetches video duration in seconds using yt-dlp."""
    print("▶ Fetching video metadata...")
    ydl_opts = {'quiet': True, 'get_duration': True, 'nocheckcertificate': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = info.get('duration')
            if duration:
                return int(duration)
    except Exception as e:
        print(f"  - Warning: Could not fetch video duration. {e}", file=sys.stderr)
    return None


def sec_to_ts(seconds: int) -> str:
    """Converts total seconds to HH:MM:SS timestamp string."""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ────────── yt-dlp phase ──────────── #

def download_audio(url: str, start: str, end: str,
                   dest_mp3: pathlib.Path):
    """Downloads and extracts the specified audio clip using yt-dlp."""
    print("[1/2] Downloading & Extracting Clip …")

    ydl_opts = {
        'format': 'bestaudio[ext=aac]/bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
        # Provide the output path template without extension
        'outtmpl': str(dest_mp3.with_suffix('')),
        'download_sections': f"*{start}-{end}",
        'nocolor': True,
        'logger': YtdlpLogger(),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            retcode = ydl.download([url])
    except Exception as e:
        raise RuntimeError(f"yt-dlp process failed with an exception: {e}") from e

    if retcode != 0:
        raise RuntimeError(f"yt-dlp process failed with exit code {retcode}. "
                           "Check the log above for ffmpeg or download errors.")
    if not dest_mp3.exists():
        raise RuntimeError("yt-dlp finished, but the final MP3 file was not found.")

    print(f"✔ Download finished. MP3 saved to: {dest_mp3}")


# ───────── transcription phase ────── #

def transcribe(mp3_path: pathlib.Path, txt_path: pathlib.Path):
    """Transcribes the given MP3 file using Whisper and saves it to a text file."""
    print("\n[2/2] Transcribing audio …")
    t0 = time.time()
    model_name = "small"

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            print(f"▶ CUDA (GPU) is available. Using device: {device}")
        else:
            print("▶ CUDA (GPU) not available. Using device: CPU.")
    except ImportError:
        print("▶ PyTorch is not installed. Using device: CPU.")

    print(f"▶ Loading Whisper model: '{model_name}' (this may download the model on first run)...")
    model = whisper.load_model(model_name, device=device)

    print("▶ Starting transcription process (Press Ctrl+C to stop and save progress)...")
    # The FP16 warning is normal on CPU and can be ignored.

    captured_output = io.StringIO()
    status = "Unknown"

    try:
        # Create a Tee stream to write to both stdout and our capture variable
        tee_stream = Tee(sys.stdout, captured_output)
        with contextlib.redirect_stdout(tee_stream):
            result = model.transcribe(str(mp3_path), verbose=True)

        # This block runs if transcription completes normally
        print("\n▶ Transcription complete. Writing full transcript to file...")
        transcript_text = "\n".join(seg["text"].strip() for seg in result["segments"])
        txt_path.write_text(transcript_text, encoding="utf-8")
        status = "Completed"

    except KeyboardInterrupt:
        print("\n\n▶ Keyboard interrupt detected. Saving partial transcript...")
        raw_log = captured_output.getvalue()

        # Grab just the spoken text
        lines = WHISPER_VERBOSE_RE.findall(raw_log)
        partial_transcript = "\n".join(lines) if lines else raw_log

        txt_path.write_text(partial_transcript, encoding="utf-8")
        status = "Interrupted (Partial)"
        raise


    finally:
        elapsed_time = time.time() - t0
        print(f"\n✔ Transcript ({status}) saved to: {txt_path}")
        print(f"  Transcription ran for {elapsed_time / 60:.1f} minutes ({elapsed_time:.0f} seconds).")


# ───────────── main ───────────── #

def main():
    """Main script execution block."""
    print("--- SCRIPT START ---")

    try:
        print("▶ Validating and adjusting settings...")
        duration_sec = get_video_duration(URL)

        if duration_sec:
            duration_ts = sec_to_ts(duration_sec)
            print(f"  - Video duration is {duration_ts} ({duration_sec}s).")

            # Process START_TS, defaulting to 0 if invalid
            try:
                start_sec = ts_to_sec(START_TS) if START_TS else 0
                if not START_TS:
                    print(f"  - START_TS is empty, defaulting to clip start (00:00:00).")
                elif start_sec >= duration_sec:
                    print(f"  - Warning: START_TS ({START_TS}) is past the video's end. Defaulting to clip start (00:00:00).")
                    start_sec = 0
            except ValueError:
                print(f"  - Warning: Invalid START_TS format ({START_TS!r}). Defaulting to clip start (00:00:00).")
                start_sec = 0

            # Process END_TS, defaulting to max duration if invalid
            try:
                end_sec = ts_to_sec(END_TS) if END_TS else duration_sec
                if not END_TS:
                    print(f"  - END_TS is empty, defaulting to clip end ({duration_ts}).")
                elif end_sec > duration_sec:
                    print(f"  - Info: END_TS ({END_TS}) exceeds video duration. Adjusting to clip end ({duration_ts}).")
                    end_sec = duration_sec
            except ValueError:
                print(f"  - Warning: Invalid END_TS format ({END_TS!r}). Defaulting to clip end ({duration_ts}).")
                end_sec = duration_sec

            # Final check: if the calculated range is invalid, default to the full video.
            if start_sec >= end_sec:
                print(f"  - Warning: The resulting time range is invalid (start time is not before end time).")
                print(f"  - Defaulting to the full video clip: 00:00:00 -> {duration_ts}")
                start_ts_final = "00:00:00"
                end_ts_final = duration_ts
            else:
                start_ts_final = sec_to_ts(start_sec)
                end_ts_final = sec_to_ts(end_sec)
        else:
            # Fallback if duration can't be fetched
            print("  - Warning: Could not determine video duration. Timestamps will be used as provided.")
            if not START_TS or not END_TS:
                raise ValueError("Both START_TS and END_TS must be set when video duration is unavailable.")
            if ts_to_sec(END_TS) <= ts_to_sec(START_TS):
                raise ValueError(f"Timestamp range is invalid: END_TS ({END_TS}) must be after START_TS ({START_TS}).")
            start_ts_final = START_TS
            end_ts_final = END_TS

        print(f"▶ Using 'yt-dlp' Python package.")

        base_out_dir = ensure_outdir(OUTDIR)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_out_dir / f"run_{stamp}"
        run_dir.mkdir()
        print(f"▶ Created run directory: {run_dir}")
        print(f"▶ Final clip range to download: {start_ts_final} to {end_ts_final}")


        mp3_file = run_dir / "audio.mp3"
        txt_file = run_dir / "transcript.txt"

        print("-" * 25)
        download_audio(URL, start_ts_final, end_ts_final, mp3_file)

        transcribe(mp3_file, txt_file)

        print("\n--- SCRIPT FINISHED SUCCESSFULLY ---")

    except KeyboardInterrupt:
        # The 'transcribe' function already handled saving the file and printing status
        print("\n▶ Script execution stopped by user.")
        print("--- SCRIPT END (INTERRUPTED) ---")
        sys.exit(0)
    except Exception as exc:
        print(f"\n--- SCRIPT FAILED ---", file=sys.stderr)
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()