"""
audio_mixer.py
──────────────
Assembles individual TTS audio segments into one full-length audio track,
aligned to the original video's timeline, then muxes it into the video.

Strategy (sync-accurate):
- Allocate a numpy float32 buffer exactly matching the video duration.
- For each segment, place TTS audio at the EXACT original timestamp.
  This guarantees perfect sync regardless of TTS length.
- If TTS is longer than the original slot → time-stretch with ffmpeg atempo
  (high-quality, no chipmunk effect). Capped at 2x speed.
- If TTS overflows the next segment's start → hard-trim it.
  (Better to cut a word than drift out of sync for the rest of the video.)
- Replace the video's audio track using ffmpeg stream copy (no video re-encode).
- Optionally blend background audio at reduced volume (disabled by default).
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from pydub import AudioSegment
from rich.console import Console
from rich.progress import track

console = Console()

SAMPLE_RATE  = 44100
NUM_CHANNELS = 2


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_video_duration(video_path: str) -> float:
    """Get the duration of a video file in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _segment_to_numpy(audio: AudioSegment, sample_rate: int) -> np.ndarray:
    """Convert a pydub AudioSegment to a float32 numpy array shaped (N, 2)."""
    audio = audio.set_frame_rate(sample_rate).set_channels(NUM_CHANNELS).set_sample_width(2)
    raw = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32)
    raw /= 32768.0          # normalise to [-1, 1]
    return raw.reshape(-1, NUM_CHANNELS)


def _atempo_stretch(audio: AudioSegment, speed: float, temp_dir: str) -> AudioSegment:
    """
    Time-stretch audio using ffmpeg's atempo filter.
    Higher quality than pydub speedup — no chipmunk effect.
    atempo supports 0.5–2.0, so we chain filters for speed > 2.0.
    """
    if speed <= 1.05:
        return audio   # barely any difference, skip

    in_path  = os.path.join(temp_dir, "_atempo_in.wav")
    out_path = os.path.join(temp_dir, "_atempo_out.wav")
    audio.export(in_path, format="wav")

    # Build atempo filter chain (each step limited to [0.5, 2.0])
    if speed <= 2.0:
        filter_str = f"atempo={speed:.4f}"
    else:
        # e.g. speed=3.0 → atempo=2.0,atempo=1.5
        filter_str = f"atempo=2.0,atempo={speed / 2.0:.4f}"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", in_path,
        "-filter:a", filter_str,
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not os.path.exists(out_path):
        console.print(f"[yellow]⚠  atempo failed (speed={speed:.2f}), using original[/yellow]")
        return audio

    return AudioSegment.from_wav(out_path)


# ── Main mixing function ─────────────────────────────────────────────────────

def build_dubbed_audio(
    synthesized_segments: list[dict],
    video_path: str,
    output_dir: str,
    sample_rate: int = SAMPLE_RATE,
    no_vocals_path: str | None = None,
    keep_background: bool = False,
    bg_volume_db: float = -18.0,
) -> str:
    """
    Build a full-length dubbed audio track from synthesized segments.

    Uses a numpy buffer for O(N) assembly with mathematically perfect sync:
    every segment is placed at its exact original start timestamp.

    Args:
        synthesized_segments: Output from synthesizer.synthesize_segments().
        video_path:           Path to the original video (to get total duration).
        output_dir:           Directory to save the merged audio file.
        sample_rate:          Output sample rate in Hz.
        no_vocals_path:       Path to Demucs no_vocals.wav (BGM + SFX + laughing).
                              If provided, used as the base canvas — gives "same
                              energy" feel with original music and laughter intact.
        keep_background:      DEPRECATED. Use no_vocals_path instead.
                              If True AND no_vocals_path is None, blends the raw
                              original audio at reduced volume (more Hindi bleed).
        bg_volume_db:         Volume for background track in dB (default −18dB).

    Returns:
        Path to the final merged audio WAV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = str(output_dir / "dubbed_audio.wav")

    console.print("[bold cyan]🎚  Building dubbed audio track...[/bold cyan]")

    video_duration_s  = _get_video_duration(video_path)
    total_samples     = int(video_duration_s * sample_rate)
    console.print(f"[dim]Video duration: {video_duration_s:.1f}s → {total_samples:,} samples[/dim]")

    # ── Allocate output buffer ────────────────────────────────────────────────
    # float32, shape (total_samples, 2) — one column per channel
    output_buf = np.zeros((total_samples, NUM_CHANNELS), dtype=np.float32)

    # ── Build base canvas ─────────────────────────────────────────────────────
    if no_vocals_path and Path(no_vocals_path).exists():
        # Best path: use the Demucs-separated background track.
        # This contains original BGM, laughing, applause, and SFX — NO Hindi speech.
        console.print("[bold green]🎵 Using Demucs background track (music/SFX/laughing preserved!)[/bold green]")
        try:
            bg = AudioSegment.from_wav(no_vocals_path)
            bg = bg.set_frame_rate(sample_rate).set_channels(NUM_CHANNELS)
            bg = bg.apply_gain(bg_volume_db)
            bg_np  = _segment_to_numpy(bg, sample_rate)
            bg_len = min(len(bg_np), total_samples)
            output_buf[:bg_len] = bg_np[:bg_len]
            console.print(f"[green]✓ Background track loaded at {bg_volume_db}dB[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠  Could not load background track: {e}[/yellow]")

    elif keep_background:
        # Legacy path: blend the raw original audio at low volume.
        # Has minor Hindi vocal bleed-through since we didn’t use Demucs.
        console.print(f"[dim]Extracting raw background audio at {bg_volume_db}dB...[/dim]")
        bg_wav = str(output_dir / "bg_audio.wav")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_path, "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate), "-ac", str(NUM_CHANNELS),
            bg_wav,
        ], capture_output=True)
        if Path(bg_wav).exists():
            try:
                bg = AudioSegment.from_wav(bg_wav).apply_gain(bg_volume_db)
                bg_np  = _segment_to_numpy(bg, sample_rate)
                bg_len = min(len(bg_np), total_samples)
                output_buf[:bg_len] += bg_np[:bg_len]
                console.print(f"[green]✓ Raw background blended at {bg_volume_db}dB[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠  Background audio failed: {e}[/yellow]")
    else:
        console.print("[dim]Using silence as base (no background audio)[/dim]")

    # ── Sort segments by original start time ─────────────────────────────────
    sorted_segs = sorted(synthesized_segments, key=lambda x: x["start"])

    # Pre-compute next-segment start times for trim boundary
    next_starts = [
        int(sorted_segs[i + 1]["start"] * sample_rate) if i + 1 < len(sorted_segs) else total_samples
        for i in range(len(sorted_segs))
    ]

    placed_count  = 0
    skipped_count = 0
    stretched_count = 0

    # Use a temp dir for atempo intermediate files
    with tempfile.TemporaryDirectory() as tmp:
        for i, seg in enumerate(track(sorted_segs, description="Mixing segments...", console=console)):
            try:
                tts_audio = AudioSegment.from_mp3(seg["audio_path"])
            except Exception as e:
                console.print(f"[red]⚠  Cannot load segment {seg['index']}: {e}[/red]")
                skipped_count += 1
                continue

            tts_audio = tts_audio.set_frame_rate(sample_rate).set_channels(NUM_CHANNELS)

            # Original slot timing
            slot_start_s    = seg["start"]
            slot_end_s      = seg["end"]
            slot_duration_s = max(slot_end_s - slot_start_s, 0.1)
            tts_duration_s  = len(tts_audio) / 1000.0

            # Time-stretch if TTS is longer than the slot
            if tts_duration_s > slot_duration_s:
                speed = min(tts_duration_s / slot_duration_s, 2.0)
                if speed > 1.05:
                    tts_audio = _atempo_stretch(tts_audio, speed, tmp)
                    stretched_count += 1

            # Convert to numpy
            tts_np = _segment_to_numpy(tts_audio, sample_rate)

            # Exact placement position in samples
            start_sample = int(slot_start_s * sample_rate)
            # Hard trim: never write past next segment's start or end of video
            max_end_sample = min(next_starts[i], total_samples)
            end_sample     = min(start_sample + len(tts_np), max_end_sample)
            write_len      = end_sample - start_sample

            if write_len <= 0 or start_sample >= total_samples:
                skipped_count += 1
                continue

            # Overlay (add) the TTS into the buffer
            output_buf[start_sample:end_sample] += tts_np[:write_len]
            placed_count += 1

    # ── Normalise to prevent clipping ────────────────────────────────────────
    peak = np.max(np.abs(output_buf))
    if peak > 1.0:
        output_buf /= peak

    console.print(
        f"[green]✓ Placed {placed_count} segments "
        f"({stretched_count} time-stretched, {skipped_count} skipped)[/green]"
    )

    # ── Export as WAV ─────────────────────────────────────────────────────────
    console.print("[bold cyan]💾 Exporting dubbed audio...[/bold cyan]")

    # Convert float32 → int16 PCM
    out_int16 = (output_buf * 32767).astype(np.int16)
    audio_out = AudioSegment(
        out_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,    # 16-bit
        channels=NUM_CHANNELS,
    )
    audio_out.export(merged_path, format="wav")
    console.print(f"[green]✓ Dubbed audio saved:[/green] {merged_path}")
    return merged_path


# ── Mux into video ───────────────────────────────────────────────────────────

def mux_audio_into_video(
    video_path: str,
    dubbed_audio_path: str,
    output_path: str,
) -> str:
    """
    Replace the audio track in the video with the dubbed audio using ffmpeg.
    Video stream is stream-copied (no re-encode).
    Audio is encoded to AAC for MP4 compatibility.

    Args:
        video_path:        Path to the original video.
        dubbed_audio_path: Path to the dubbed WAV audio.
        output_path:       Path for the final dubbed video.

    Returns:
        Path to the output dubbed video.
    """
    console.print("[bold cyan]🎬 Muxing dubbed audio into video (stream copy)...[/bold cyan]")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", dubbed_audio_path,
        "-c:v", "copy",       # Copy video stream — no re-encode
        "-c:a", "aac",        # Encode audio to AAC (MP4-compatible)
        "-b:a", "192k",
        "-map", "0:v:0",      # Video from first input
        "-map", "1:a:0",      # Audio from second input (our dub)
        # NOTE: Do NOT use -shortest — we want the full video duration preserved.
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed:\n{result.stderr}")

    console.print(f"[green]✓ Final dubbed video saved:[/green] {output_path}")
    return output_path
