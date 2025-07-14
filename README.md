# link/local clip → MP3 → Whisper Transcript

A customizable, all-in-one script to download and clip from a URL/local file and generate a timestamped transcript using faster-whisper. I will add proper command-line argument handling soon.

**Requirements**

* Python ≥ 3.11
* ffmpeg on PATH
* `pip install -r requirements.txt`
* For GPU, install PyTorch with CUDA

**Usage**

1. Edit `SOURCE`, `START_TS`, `END_TS`, `OUTDIR` in `script.py` and run via IDE.
2. OR Run:

   ```bash
   python script.py "SOURCE_URL_OR_PATH"
   ```
3. Find `audio.mp3` and `transcript.txt` in the output folder.

**(Work in progress)**
