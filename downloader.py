"""
downloader.py
─────────────
Downloads a YouTube video using yt-dlp and extracts its audio as a WAV file.

Note: Audio is extracted at 44100Hz stereo (full quality) so that:
  - Demucs gets the best possible input for vocal/BGM separation.
  - XTTS gets high-quality speaker reference clips.
  - Whisper still works correctly (faster-whisper resamples internally).
"""

import os
import sys
import subprocess
from pathlib import Path
from rich.console import Console

console = Console()


def download_video(url: str, output_dir: str) -> tuple[str, str]:
    """
    Download a YouTube video and extract its audio.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the downloaded files.

    Returns:
        A tuple of (video_path, audio_path) as strings.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_template = str(output_dir / "source_video.%(ext)s")
    audio_path = str(output_dir / "source_audio.wav")

    # Fast-path: skip download if files already exist
    existing_video = output_dir / "source_video.mp4"
    if existing_video.exists() and Path(audio_path).exists():
        console.print(f"[yellow]⚡ Skipping download — files already exist[/yellow]")
        console.print(f"[green]✓ Using cached video:[/green] {existing_video}")
        console.print(f"[green]✓ Using cached audio:[/green] {audio_path}")
        return str(existing_video), audio_path

    console.print("[bold cyan]📥 Downloading video...[/bold cyan]")

    # Download the best quality video + audio merged
    download_cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--extractor-args", "youtube:player_client=android",  # Bypasses Colab bot protection
        "--output", video_template,
        "--newline",
        url,
    ]

    result = subprocess.run(download_cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed to download video from: {url}")

    # Find the downloaded video file
    video_path = None
    for f in output_dir.iterdir():
        if f.stem == "source_video" and f.suffix in (".mp4", ".mkv", ".webm", ".avi"):
            video_path = str(f)
            break

    if not video_path:
        raise FileNotFoundError("Downloaded video file not found in output directory.")

    console.print(f"[green]✓ Video saved:[/green] {video_path}")

    # Extract audio as WAV using ffmpeg
    # 44100Hz stereo: full quality for Demucs + XTTS; Whisper resamples internally.
    console.print("[bold cyan]🔊 Extracting audio track (44.1kHz stereo)...[/bold cyan]")
    extract_cmd = [
        "ffmpeg",
        "-y",                    # overwrite if exists
        "-i", video_path,
        "-vn",                   # no video
        "-acodec", "pcm_s16le",  # standard WAV PCM
        "-ar", "44100",          # full quality — Demucs & XTTS need this
        "-ac", "2",              # stereo
        audio_path,
    ]

    result = subprocess.run(extract_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")

    console.print(f"[green]✓ Audio extracted:[/green] {audio_path}")
    return video_path, audio_path
