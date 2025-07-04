# link/local clip → MP3 → Whisper Transcript

Extract a segment from a local file or URL to MP3 and generate a Whisper transcript.

**Requirements**

* Python ≥ 3.11
* ffmpeg on PATH
* `pip install -U yt-dlp git+https://github.com/openai/whisper.git`

**Usage**

1. Edit `SOURCE`, `START_TS`, `END_TS`, `OUTDIR` in `script.py` (or pass a link/file path as CLI arg).
2. Run:

   ```bash
   python script.py [SOURCE]
   ```
3. Find `audio.mp3` and `transcript.txt` in the output folder.

**(Work in progress)**
