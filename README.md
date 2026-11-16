# YouTube Clipper

A local web app for finding speech-based highlights and exporting vertical Shorts.

## Windows setup

Install Python 3.11 or 3.12, FFmpeg with subtitle and libx264 support, and Deno for YouTube imports. Make sure `py`, `ffmpeg`, `ffprobe`, and `deno` are on PATH. Open PowerShell in the project folder and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

If using Python 3.11, replace `-3.12` with `-3.11`. Open http://127.0.0.1:8765.

Upload an MP4 or paste a YouTube video URL for footage you have permission to reuse. For the first run, allow time for faster-whisper to download the model. The app uses CPU transcription by default. If your system already has compatible CUDA 12 and cuDNN 9 libraries on PATH, set `$env:CLIPPER_DEVICE = "cuda"` before starting the app to enable GPU transcription. To update YouTube import support, run `.\.venv\Scripts\python.exe -m pip install -U "yt-dlp[default]"`. If YouTube rejects an import, upload your local MP4 copy.

## Editing

The app ranks transcript ranges using opening cues, sentence endings, and speech density. It suggests up to five distinct excerpts, usually 20–60 seconds. Review them in the source preview, adjust start and end times, and edit the caption text if needed. Choose **Fit entire frame** for wide shots or **Center crop** for footage with a centered subject. Export to review and download a captioned 1080 × 1920 H.264/AAC MP4.

Suggestions use speech, so silent footage needs a different workflow. The framing is fixed for each export. Jobs are kept in memory while the app is running. The source files and exports remain in the `jobs` folder after you close the app. Delete old job folders when the app is stopped to recover disk space.
