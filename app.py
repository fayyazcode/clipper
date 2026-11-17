import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "jobs"
JOBS = {}
LOCK = threading.Lock()
WORKER = ThreadPoolExecutor(max_workers=1)
MODEL = None
LIMIT = 2 * 1024**3


def run(args, folder):
    if args[0] in {"ffmpeg", "ffprobe"}:
        executable = shutil.which(args[0])
        if executable is None:
            raise RuntimeError(f"{args[0]} is missing. Install FFmpeg and reopen PowerShell.")
        args = [executable, *args[1:]]
    result = subprocess.run(args, cwd=folder, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[-1800:] or f"{args[0]} failed.")
    return result.stdout


def youtube_url(value):
    url = urlsplit(value)
    if url.scheme != "https":
        raise ValueError("Use an HTTPS YouTube video link.")
    if url.netloc.lower() == "youtu.be":
        video_id = url.path.strip("/")
    elif url.netloc.lower() in {"www.youtube.com", "youtube.com", "m.youtube.com"}:
        if url.path == "/watch":
            video_id = parse_qs(url.query).get("v", [""])[0]
        elif url.path.startswith(("/shorts/", "/live/")):
            video_id = url.path.split("/")[2]
        else:
            video_id = ""
    else:
        video_id = ""
    if not re.fullmatch(r"[\w-]{11}", video_id, re.ASCII):
        raise ValueError("Enter a link to one YouTube video.")
    return "https://www.youtube.com/watch?v=" + video_id


def select_clips(segments):
    candidates = []
    for index, first in enumerate(segments):
        lines = []
        for last in segments[index:]:
            length = last["end"] - first["start"]
            if length > 60:
                break
            lines.append(last["text"])
            if length < 20:
                continue
            words = re.findall(r"\w+", " ".join(lines).lower())
            opening = lines[0].lower()
            hook = bool(re.search(r"\b(why|how|what|but|never|discovered|mystery)\b", opening))
            complete = lines[-1].endswith((".", "?", "!"))
            score = 3 * hook + 2 * complete + min(len(set(words)) / length, 3)
            score -= abs(length - 35) / 20
            title = " ".join(lines[0].split()[:10]).rstrip(".,:;")
            candidates.append((score, len(candidates), {
                "start": first["start"], "end": last["end"], "title": title,
            }))
    clips = []
    for _, _, candidate in sorted(candidates, reverse=True):
        if any(candidate["start"] < clip["end"] and candidate["end"] > clip["start"]
               for clip in clips):
            continue
        clips.append(candidate)
        if len(clips) == 5:
            break
    if not clips and segments:
        start = segments[0]["start"]
        clips.append({"start": start, "end": min(start + 60, segments[-1]["end"]),
                      "title": "Opening excerpt"})
    return clips


def update(job_id, **changes):
    with LOCK:
        JOBS[job_id].update(changes)


def download_youtube(link, folder, preview):
    if preview:
        formats = (
            "b[height<=360][vcodec^=avc1][acodec^=mp4a]"
            "/bv*[height<=360][vcodec^=avc1]+ba[ext=m4a]"
            "/b[height<=480][ext=mp4]/bv*[height<=480]+ba/b[height<=480]"
        )
        filename = "source"
    else:
        formats = (
            "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]"
            "/b[height<=1080][ext=mp4]/bv*[height<=1080]+ba/b[height<=1080]"
        )
        filename = "export-source"

    with TemporaryDirectory(dir=folder) as temp:
        download_folder = Path(temp)
        run([sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist",
             "--max-filesize", "2G", "-f", formats, "--merge-output-format", "mp4",
             "--remux-video", "mp4", "-o", "video.%(ext)s", link], download_folder)
        video = download_folder / "video.mp4"
        if not video.is_file():
            raise RuntimeError("YouTube import produced no MP4. Try uploading a local copy.")
        video.replace(folder / f"{filename}.mp4")


def analyze(job_id, link):
    global MODEL
    folder = DATA / job_id
    try:
        if link:
            update(job_id, status="Downloading a lightweight preview")
            download_youtube(link, folder, preview=True)
        info = json.loads(run(["ffprobe", "-v", "error", "-show_streams",
                               "-show_format", "-of", "json", "source.mp4"], folder))
        kinds = {stream["codec_type"] for stream in info["streams"]}
        if not {"audio", "video"} <= kinds:
            raise ValueError("The source needs a video track and an audio track.")
        duration = float(info["format"]["duration"])
        update(job_id, status="Transcribing")
        if MODEL is None:
            from faster_whisper import WhisperModel
            device = os.getenv("CLIPPER_DEVICE", "cpu")
            precision = "float16" if device == "cuda" else "int8"
            MODEL = WhisperModel("small", device=device, compute_type=precision)
        try:
            transcript, _ = MODEL.transcribe(str(folder / "source.mp4"), vad_filter=True)
            segments = [{"start": segment.start, "end": min(segment.end, duration),
                         "text": segment.text.strip()} for segment in transcript
                        if segment.text.strip() and segment.start < duration]
        except RuntimeError:
            MODEL = None
            raise
        clips = select_clips(segments)
        if not clips:
            raise ValueError("No speech was detected in this video.")
        update(job_id, status="ready", duration=duration, segments=segments, clips=clips)
    except Exception as error:
        update(job_id, status="error", error=str(error))


def stamp(seconds):
    minutes, milliseconds = divmod(round(seconds * 1000), 60_000)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{milliseconds // 1000:02},{milliseconds % 1000:03}"


def render(job_id, export_id, start, end, layout, captions):
    folder = DATA / job_id
    with LOCK:
        job = JOBS[job_id]
        export = job["exports"][export_id]
        link = job["link"]
    try:
        source = "source.mp4"
        if link:
            source = "export-source.mp4"
            if not (folder / source).is_file():
                with LOCK:
                    export.update(status="Downloading high-quality source")
                download_youtube(link, folder, preview=False)
        with LOCK:
            export.update(status="Rendering")
        if layout in {"reel", "reel-zoom"}:
            foreground = "scale=1080:1920:force_original_aspect_ratio=decrease:force_divisible_by=2"
            if layout == "reel-zoom":
                foreground += (
                    ",scale=w='2*trunc(iw*1.15/2)':h='2*trunc(ih*1.15/2)'"
                    ",crop=w='min(iw,1080)':h='min(ih,1920)'"
                )
            video_filter = (
                "split[background][foreground];"
                "[background]scale=180:320:force_original_aspect_ratio=increase,"
                "crop=180:320,gblur=sigma=12,eq=brightness=-0.17,"
                "scale=1080:1920,setsar=1[blurred];"
                f"[foreground]{foreground},setsar=1[sharp];"
                "[blurred][sharp]overlay=(W-w)/2:(H-h)/2"
            )
        elif layout == "crop":
            video_filter = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
        else:
            video_filter = "scale=1080:1920:force_original_aspect_ratio=decrease:force_divisible_by=2,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
        video_filter += ",setsar=1"
        cues = []
        for caption in captions:
            cue_start, cue_end = max(start, caption["start"]), min(end, caption["end"])
            line = " ".join(caption["text"].split()).replace("<", "‹").replace(">", "›")
            if line and cue_end - cue_start >= 0.01:
                cues.append(f"{len(cues) + 1}\n{stamp(cue_start-start)} --> {stamp(cue_end-start)}\n{line}")
        if cues:
            subtitle = f"{export_id}.srt"
            (folder / subtitle).write_text("\n\n".join(cues) + "\n", encoding="utf-8")
            video_filter += f",subtitles={subtitle}:force_style='FontSize=14,Outline=2,MarginV=36'"
        output = export_id + ".mp4"
        if layout in {"reel", "reel-zoom"}:
            video_args = ["-filter_complex", f"[0:v:0]{video_filter}[video]", "-map", "[video]"]
        else:
            video_args = ["-vf", video_filter, "-map", "0:v:0"]
        run(["ffmpeg", "-nostdin", "-v", "error", "-ss", str(start), "-i", source,
             "-t", str(end-start), *video_args, "-map", "0:a:0",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", output], folder)
        with LOCK:
            export.update(status="ready", url=f"/media/{job_id}/{output}")
    except Exception as error:
        with LOCK:
            export.update(status="error", error=str(error))


class Handler(BaseHTTPRequestHandler):
    def reply(self, data, status=200, kind="application/json"):
        if kind == "application/json":
            data = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def media(self, path, download):
        length = path.stat().st_size
        start, end = 0, length - 1
        interval = self.headers.get("Range")
        if interval:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", interval)
            if match and match[1]:
                start = int(match[1])
                end = min(end, int(match[2])) if match[2] else end
            elif match and match[2]:
                start = max(0, length - int(match[2]))
            else:
                start = length
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{length}")
                self.end_headers()
                return
        self.send_response(206 if interval else 200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(end-start+1))
        if interval:
            self.send_header("Content-Range", f"bytes {start}-{end}/{length}")
        if download:
            self.send_header("Content-Disposition", 'attachment; filename="youtube-short.mp4"')
        self.end_headers()
        with path.open("rb") as source:
            source.seek(start)
            remaining = end-start+1
            try:
                while remaining:
                    block = source.read(min(remaining, 1024**2))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def valid_host(self):
        host = self.headers.get("Host", "")
        return host in {"localhost", "127.0.0.1",
                        f"localhost:{self.server.server_port}",
                        f"127.0.0.1:{self.server.server_port}"}

    def do_GET(self):
        if not self.valid_host():
            self.reply({"error": "Open the app on localhost."}, 403)
            return
        address = urlsplit(self.path)
        if address.path == "/":
            self.reply((ROOT / "index.html").read_bytes(), kind="text/html; charset=utf-8")
            return
        match = re.fullmatch(r"/jobs/([a-f0-9]{32})", address.path)
        if match:
            with LOCK:
                job = JOBS.get(match[1])
                data = json.loads(json.dumps(job)) if job else None
            if data:
                self.reply(data)
                return
        match = re.fullmatch(r"/media/([a-f0-9]{32})/(source|[a-f0-9]{32})\.mp4", address.path)
        if match:
            path = DATA / match[1] / f"{match[2]}.mp4"
            if path.is_file():
                self.media(path, "download" in parse_qs(address.query))
                return
        self.reply({"error": "Item not found."}, 404)

    def read_json(self, length):
        if length > 2 * 1024**2 or self.headers.get_content_type() != "application/json":
            raise ValueError("Expected a JSON request.")
        return json.loads(self.rfile.read(length))

    def do_POST(self):
        origin = self.headers.get("Origin")
        if not self.valid_host() or (origin and urlsplit(origin).netloc != self.headers.get("Host")):
            self.reply({"error": "Open the app on localhost."}, 403)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= LIMIT:
                raise ValueError("Choose a source under 2 GiB.")
            if self.path in {"/upload", "/import"}:
                link = youtube_url(self.read_json(length)["url"]) if self.path == "/import" else None
                if not link and self.headers.get_content_type() != "application/octet-stream":
                    raise ValueError("Choose an MP4 file.")
                job_id = uuid.uuid4().hex
                folder = DATA / job_id
                folder.mkdir(parents=True)
                if not link:
                    try:
                        with (folder / "source.mp4").open("wb") as destination:
                            remaining = length
                            while remaining:
                                block = self.rfile.read(min(remaining, 1024**2))
                                if not block:
                                    raise ValueError("The upload ended early.")
                                destination.write(block)
                                remaining -= len(block)
                    except Exception:
                        (folder / "source.mp4").unlink(missing_ok=True)
                        folder.rmdir()
                        raise
                with LOCK:
                    existing = next((key for key, job in JOBS.items()
                                     if link and job["link"] == link and job["status"] != "error"
                                     and (DATA / key).is_dir()), None)
                    if not existing:
                        JOBS[job_id] = {"status": "Queued", "exports": {}, "link": link,
                                        "source": f"/media/{job_id}/source.mp4"}
                if existing:
                    folder.rmdir()
                else:
                    WORKER.submit(analyze, job_id, link)
                self.reply({"id": existing or job_id}, 202)
                return
            match = re.fullmatch(r"/jobs/([a-f0-9]{32})/export", self.path)
            if match:
                request = self.read_json(length)
                start, end = float(request["start"]), float(request["end"])
                layout = request["layout"]
                if layout not in {"crop", "pad", "reel", "reel-zoom"} or not math.isfinite(start) or not math.isfinite(end):
                    raise ValueError("Check the trim points and framing.")
                captions = []
                for cue in request["captions"]:
                    cue_start, cue_end = float(cue["start"]), float(cue["end"])
                    if not math.isfinite(cue_start) or not math.isfinite(cue_end) or cue_end <= cue_start:
                        raise ValueError("Check the caption timing.")
                    captions.append({"start": cue_start, "end": cue_end, "text": str(cue["text"])})
                with LOCK:
                    job = JOBS.get(match[1])
                    if not job or job["status"] != "ready":
                        raise ValueError("Wait for analysis to finish.")
                    if not 0 <= start < end <= job["duration"] or not 1 <= end-start <= 90:
                        raise ValueError("Choose a clip between 1 and 90 seconds within the source.")
                    export_id = uuid.uuid4().hex
                    job["exports"][export_id] = {"status": "Queued or rendering"}
                WORKER.submit(render, match[1], export_id, start, end, layout, captions)
                self.reply({"id": export_id}, 202)
                return
            self.reply({"error": "Unknown action."}, 404)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            self.reply({"error": str(error)}, 400)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Open http://127.0.0.1:8765", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
