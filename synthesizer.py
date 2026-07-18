"""
synthesizer.py
──────────────
Converts translated English text into natural-sounding speech.

Two synthesis modes:

1. EDGE-TTS MODE (default, fast):
   - Uses Microsoft Edge TTS cloud API.
   - 15 concurrent async requests → 1500 segments in ~2 minutes.
   - Two-voice speaker detection: alternates male/female on silence gaps.
   - Adaptive TTS rate: pre-compresses verbose segments to reduce post-processing.

2. XTTS VOICE-CLONING MODE (--clone-voice, best quality):
   - Uses Coqui XTTS-v2, a zero-shot voice cloning model.
   - For each segment, extracts the original speaker's voice slice from
     Demucs-separated vocals.wav as a 3+ second reference clip.
   - Clones the exact voice, tone, and emotional energy of the original speaker.
   - Runs on GPU; speed depends on hardware (~1x real-time on T4, ~4x on A100).
   - Best used on Lightning AI / RunPod with A10G or A100 GPU.
"""

import asyncio
import os
import tempfile
from pathlib import Path

from pydub import AudioSegment
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, track

console = Console()

# ── Edge TTS voice table ─────────────────────────────────────────────────────
VOICES = {
    "default":    "en-US-AndrewMultilingualNeural",
    "male":       "en-US-AndrewMultilingualNeural",
    "female":     "en-US-AvaMultilingualNeural",
    "male_alt":   "en-GB-RyanNeural",
    "female_alt": "en-GB-SoniaNeural",
    "neutral":    "en-US-EmmaMultilingualNeural",
}

# Two-voice pair for speaker alternation in Edge-TTS mode
SPEAKER_VOICES = [
    "en-US-AndrewMultilingualNeural",   # Speaker A – male
    "en-US-AvaMultilingualNeural",      # Speaker B – female
]

# Silence gap (seconds) that triggers a speaker change
SPEAKER_CHANGE_GAP_S = 2.5

# Minimum reference clip duration for XTTS (seconds)
XTTS_MIN_REF_DURATION_S = 3.0


# ── Adaptive TTS rate (Edge-TTS mode only) ───────────────────────────────────

def _adaptive_rate(original_duration_s: float, text: str) -> str:
    """Estimate needed TTS speaking rate to avoid post-processing over-compression."""
    word_count = len(text.split())
    if original_duration_s <= 0:
        return "+0%"
    estimated_tts_s = word_count / 2.5
    ratio = estimated_tts_s / original_duration_s
    if ratio > 1.6:
        return "+30%"
    elif ratio > 1.3:
        return "+20%"
    elif ratio > 1.1:
        return "+10%"
    return "+0%"


# ── Edge-TTS: async batch synthesis ──────────────────────────────────────────

async def _edge_synthesize_one(text: str, output_path: str, voice: str, rate: str) -> bool:
    """Async synthesize one segment via Edge TTS."""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(output_path)
        return True
    except Exception as e:
        console.print(f"[red]⚠  TTS error: {e}[/red]")
        return False


def _synthesize_edge_tts(
    segments: list,
    output_dir: Path,
    voice_name: str,
    speaker_idx: list[int],
) -> list[dict]:
    """Run all segments through Edge TTS with async concurrency."""
    voice_pair = SPEAKER_VOICES.copy()
    if voice_name in ("female", "female_alt", "neutral"):
        voice_pair = [SPEAKER_VOICES[1], SPEAKER_VOICES[0]]

    primary_voice = VOICES.get(voice_name, voice_name)
    console.print(f"[bold cyan]🗣  Edge-TTS: primary voice = {primary_voice}[/bold cyan]")
    console.print(
        f"[dim]Two-voice mode: Speaker A = {voice_pair[0].split('-')[2]}, "
        f"Speaker B = {voice_pair[1].split('-')[2]}[/dim]"
    )

    async def process_one(seg, index, sem):
        if not seg.text.strip():
            return None
        mp3_path = str(output_dir / f"seg_{index:05d}.mp3")
        voice    = voice_pair[speaker_idx[index]]
        rate     = _adaptive_rate(seg.end - seg.start, seg.text)

        async with sem:
            for _ in range(3):
                ok = await _edge_synthesize_one(seg.text, mp3_path, voice, rate)
                if ok and os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                    break

        if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
            console.print(f"[red]⚠  Skipping segment {index} after 3 failed attempts[/red]")
            return None

        try:
            audio = AudioSegment.from_mp3(mp3_path)
            tts_dur = len(audio) / 1000.0
        except Exception:
            tts_dur = seg.end - seg.start

        return {
            "index": index, "start": seg.start, "end": seg.end,
            "original_duration": seg.end - seg.start,
            "tts_duration": tts_dur, "audio_path": mp3_path, "text": seg.text,
        }

    async def run_all():
        sem = asyncio.Semaphore(15)
        tasks = [process_one(seg, i, sem) for i, seg in enumerate(segments)]
        results = []
        for coro in track(
            asyncio.as_completed(tasks), total=len(tasks),
            description="Synthesizing TTS...", console=console,
        ):
            res = await coro
            if res:
                results.append(res)
        return results

    results = asyncio.run(run_all())
    return sorted(results, key=lambda x: x["index"])


# ── XTTS: voice-cloning synthesis ────────────────────────────────────────────

def _extract_reference_clip(
    vocals: AudioSegment,
    start_s: float,
    end_s: float,
    temp_dir: str,
    index: int,
) -> str:
    """
    Extract a reference voice clip for XTTS from the Demucs vocals track.

    Ensures the clip is at least XTTS_MIN_REF_DURATION_S long by expanding
    context around the segment's midpoint.
    """
    total_ms  = len(vocals)
    start_ms  = int(start_s * 1000)
    end_ms    = int(end_s   * 1000)
    duration  = (end_ms - start_ms) / 1000.0

    if duration < XTTS_MIN_REF_DURATION_S:
        mid_ms       = (start_ms + end_ms) // 2
        half_target  = int(XTTS_MIN_REF_DURATION_S * 1000) // 2
        start_ms     = max(0, mid_ms - half_target)
        end_ms       = min(total_ms, mid_ms + half_target)

    clip      = vocals[start_ms:end_ms]
    ref_path  = os.path.join(temp_dir, f"ref_{index:05d}.wav")
    clip.export(ref_path, format="wav")
    return ref_path


def _synthesize_xtts(
    segments: list,
    output_dir: Path,
    vocals_path: str,
) -> list[dict]:
    """
    Synthesize all segments using Coqui XTTS-v2 voice cloning.

    For each segment we extract the corresponding speaker slice from the
    Demucs-separated vocals track and use it as the XTTS reference voice.
    This achieves zero-shot voice cloning without any diarization library.
    """
    import torch
    from TTS.api import TTS as CoquiTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold cyan]🎙  Loading Coqui XTTS-v2 on {device}...[/bold cyan]")
    console.print("[dim]  This downloads ~1.8GB the first time (cached afterwards)[/dim]")

    tts_model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    console.print("[green]✓ XTTS-v2 loaded[/green]")

    # Load Demucs vocals for reference extraction
    console.print("[dim]  Loading vocals track for speaker reference extraction...[/dim]")
    vocals = AudioSegment.from_wav(vocals_path)

    results = []
    with tempfile.TemporaryDirectory() as ref_tmp:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Cloning voices (XTTS)...", total=len(segments))

            for i, seg in enumerate(segments):
                if not seg.text.strip():
                    progress.advance(task)
                    continue

                wav_path = str(output_dir / f"seg_{i:05d}.wav")

                # Extract reference clip for this speaker at this timestamp
                ref_path = _extract_reference_clip(
                    vocals, seg.start, seg.end, ref_tmp, i
                )

                try:
                    tts_model.tts_to_file(
                        text=seg.text,
                        speaker_wav=ref_path,
                        language="en",
                        file_path=wav_path,
                    )
                except Exception as e:
                    console.print(f"[red]⚠  XTTS failed for segment {i}: {e}[/red]")
                    progress.advance(task)
                    continue

                if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
                    progress.advance(task)
                    continue

                try:
                    audio    = AudioSegment.from_wav(wav_path)
                    tts_dur  = len(audio) / 1000.0
                except Exception:
                    tts_dur  = seg.end - seg.start

                results.append({
                    "index":             i,
                    "start":             seg.start,
                    "end":               seg.end,
                    "original_duration": seg.end - seg.start,
                    "tts_duration":      tts_dur,
                    "audio_path":        wav_path,
                    "text":              seg.text,
                })
                progress.advance(task)

    return sorted(results, key=lambda x: x["index"])


# ── Public entry point ────────────────────────────────────────────────────────

def synthesize_segments(
    segments: list,
    output_dir: str,
    voice_name: str = "default",
    clone_voice: bool = False,
    vocals_path: str | None = None,
) -> list[dict]:
    """
    Synthesize all translated segments into individual audio files.

    Args:
        segments:    List of translated Segment objects.
        output_dir:  Directory to save synthesized audio files.
        voice_name:  Key into VOICES dict (used in Edge-TTS mode).
        clone_voice: If True, use Coqui XTTS-v2 voice cloning.
                     Requires vocals_path from Demucs separation.
        vocals_path: Path to Demucs-separated vocals.wav.
                     Required when clone_voice=True.

    Returns:
        List of dicts: {index, start, end, original_duration, tts_duration,
                        audio_path, text}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if clone_voice:
        if not vocals_path or not Path(vocals_path).exists():
            console.print(
                "[yellow]⚠  --clone-voice requested but no vocals.wav found "
                "(Demucs may have failed). Falling back to Edge-TTS.[/yellow]"
            )
            clone_voice = False

    if clone_voice:
        console.print(
            "[bold green]🎭 Voice-cloning mode (Coqui XTTS-v2 + Demucs reference)[/bold green]"
        )
        synthesized = _synthesize_xtts(segments, output_dir, vocals_path)
    else:
        # Compute speaker index for two-voice alternation
        speaker_idx = [0] * len(segments)
        current_speaker = 0
        for i in range(1, len(segments)):
            gap = segments[i].start - segments[i - 1].end
            if gap >= SPEAKER_CHANGE_GAP_S:
                current_speaker = 1 - current_speaker
            speaker_idx[i] = current_speaker

        synthesized = _synthesize_edge_tts(segments, output_dir, voice_name, speaker_idx)

    console.print(f"[green]✓ Synthesized {len(synthesized)} audio segments[/green]")
    return synthesized
