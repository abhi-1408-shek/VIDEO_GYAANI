"""
transcriber.py
──────────────
Transcribes an audio file into timestamped segments using faster-whisper.
Detects the source language automatically. Falls back to CPU if CUDA
libraries (libcublas) are not available in the current environment.
"""

from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel
from rich.console import Console

console = Console()


@dataclass
class Segment:
    """A single transcribed segment with timing info."""
    start: float   # seconds
    end: float     # seconds
    text: str
    language: str  # detected source language


def transcribe(
    audio_path: str,
    model_size: str = "medium",
    device: str = "auto",
    language: str = None,
) -> list:
    """
    Transcribe audio and return a list of timestamped segments.

    Args:
        audio_path: Path to the WAV audio file.
        model_size: Whisper model size: tiny, base, small, medium, large-v2, large-v3.
        device:     'cuda', 'cpu', or 'auto' (auto-detects GPU).
        language:   Force source language (e.g. 'hi', 'de'). None = auto-detect.

    Returns:
        List of Segment objects with start, end, text, and detected language.
    """
    audio_path = str(audio_path)
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # ── Resolve device ────────────────────────────────────────────────────────
    import torch
    if device == "auto":
        compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        compute_device = device

    compute_type = "float16" if compute_device == "cuda" else "int8"

    # ── Load model with CUDA→CPU fallback ────────────────────────────────────
    console.print(f"[bold cyan]🎙  Loading Whisper model ({model_size}) on {compute_device}...[/bold cyan]")
    try:
        model = WhisperModel(model_size, device=compute_device, compute_type=compute_type)
    except Exception as e:
        err = str(e).lower()
        if compute_device == "cuda" and any(k in err for k in ("cublas", "cuda", "library", "found")):
            console.print(f"[yellow]⚠  CUDA load failed ({e}).[/yellow]")
            console.print("[yellow]   Falling back to CPU (int8)...[/yellow]")
            compute_device = "cpu"
            compute_type = "int8"
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
        else:
            raise

    console.print("[bold cyan]📝 Transcribing audio (this may take a while)...[/bold cyan]")

    # ── Shared transcribe helper (retries on CPU if CUDA runtime error) ───────
    def safe_transcribe(**kwargs):
        nonlocal model, compute_device
        try:
            return model.transcribe(audio_path, **kwargs)
        except RuntimeError as exc:
            err = str(exc).lower()
            if compute_device == "cuda" and any(k in err for k in ("cublas", "cuda", "library")):
                console.print(f"[yellow]⚠  CUDA runtime error. Switching to CPU...[/yellow]")
                model = WhisperModel(model_size, device="cpu", compute_type="int8")
                compute_device = "cpu"
                return model.transcribe(audio_path, **kwargs)
            raise

    # ── First pass: WITH VAD filter ───────────────────────────────────────────
    segments_gen, info = safe_transcribe(
        beam_size=5,
        language=language,
        word_timestamps=False,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 200,
            "min_speech_duration_ms": 50,
            "threshold": 0.25,
            "speech_pad_ms": 100,
        },
    )

    detected_lang = info.language
    console.print(f"[green]✓ Detected language:[/green] [bold]{detected_lang}[/bold] "
                  f"(confidence: {info.language_probability:.1%})")

    segments: list[Segment] = []
    count = 0
    for seg in segments_gen:
        text = seg.text.strip()
        if text:
            segments.append(Segment(
                start=seg.start,
                end=seg.end,
                text=text,
                language=detected_lang,
            ))
        count += 1
        if count % 10 == 0:
            console.print(f"[dim]  ...{count} raw chunks processed (last: {seg.end:.1f}s)[/dim]")

    console.print(f"[dim]  VAD pass yielded {len(segments)} segments[/dim]")

    # ── If VAD found very few segments, retry WITHOUT VAD ────────────────────
    if len(segments) < 5:
        console.print("[yellow]⚠  Too few segments with VAD — retrying without VAD filter...[/yellow]")
        segments_gen2, _ = safe_transcribe(
            beam_size=5,
            language=language,
            word_timestamps=False,
            vad_filter=False,
            condition_on_previous_text=True,
        )

        segments2: list[Segment] = []
        for seg in segments_gen2:
            text = seg.text.strip()
            if text:
                segments2.append(Segment(
                    start=seg.start,
                    end=seg.end,
                    text=text,
                    language=detected_lang,
                ))

        console.print(f"[dim]  No-VAD pass yielded {len(segments2)} segments[/dim]")

        if len(segments2) > len(segments):
            segments = segments2
            console.print(f"[green]✓ Using no-VAD transcription ({len(segments)} segments)[/green]")
        else:
            console.print(f"[green]✓ Using VAD transcription ({len(segments)} segments)[/green]")

    # ── Deduplicate: Whisper sometimes repeats the same text ─────────────────
    deduped = []
    prev_text = ""
    for seg in segments:
        if seg.text == prev_text:
            continue
        deduped.append(seg)
        prev_text = seg.text

    if len(deduped) < len(segments):
        console.print(f"[yellow]⚠  Removed {len(segments) - len(deduped)} duplicate segments[/yellow]")
    segments = deduped

    console.print(f"[green]✓ Transcription complete:[/green] {len(segments)} segments")
    return segments
