"""
main.py
───────
Automated Video Dubbing System – CLI entry point.

Usage:
    python main.py <YOUTUBE_URL> [options]

Examples:
    # Fast mode (Edge-TTS, 2-voice, ~8 min for 30-min video on T4)
    python main.py "https://www.youtube.com/watch?v=XYZ" --model large-v2

    # Voice-cloning mode (Coqui XTTS-v2, best quality, needs A10G/A100)
    python main.py "https://www.youtube.com/watch?v=XYZ" --model large-v2 --clone-voice
"""

import os
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import click
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║           🎬  VIDEO GYAANI  🎬             ║
║     Transcribe · Translate · Synthesize · Clone · Dub        ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_step(step_num: int, total: int, title: str):
    console.print()
    console.print(Rule(f"[bold white]Step {step_num}/{total}: {title}[/bold white]", style="bright_blue"))


def print_summary(
    url: str,
    video_path: str,
    output_path: str,
    source_lang: str,
    num_segments: int,
    elapsed: float,
    mode: str,
):
    """Print a rich summary table at the end."""
    console.print()
    table = Table(title="✅ Dubbing Complete", box=box.ROUNDED, border_style="green")
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Source URL",      url)
    table.add_row("Source Language", source_lang.upper())
    table.add_row("Segments Dubbed", str(num_segments))
    table.add_row("Synthesis Mode",  mode)
    table.add_row("Processing Time", str(timedelta(seconds=int(elapsed))))
    table.add_row("Input Video",     video_path)
    table.add_row("Output Video",    output_path)

    console.print(table)
    console.print()
    console.print(Panel(
        f"[bold green]🎉 Your dubbed video is ready![/bold green]\n"
        f"[dim]→ {output_path}[/dim]",
        border_style="green",
    ))


@click.command()
@click.argument("url")
@click.option(
    "--output", "-o",
    default="./output",
    show_default=True,
    help="Output directory for the dubbed video and intermediate files.",
)
@click.option(
    "--model", "-m",
    default="medium",
    show_default=True,
    type=click.Choice(["tiny", "base", "small", "medium", "large-v2", "large-v3"], case_sensitive=False),
    help="Whisper model size. Larger = slower but more accurate.",
)
@click.option(
    "--voice", "-v",
    default="default",
    show_default=True,
    type=click.Choice(["default", "male", "female", "male_alt", "female_alt", "neutral"], case_sensitive=False),
    help="Edge-TTS voice (ignored when --clone-voice is set).",
)
@click.option(
    "--lang", "-l",
    default=None,
    help="Force source language code (e.g. 'hi', 'de'). Default: auto-detect.",
)
@click.option(
    "--clone-voice",
    is_flag=True,
    default=False,
    help=(
        "Enable Coqui XTTS-v2 voice cloning. "
        "Clones the original speaker's voice per-segment using Demucs reference clips. "
        "Best quality but GPU-intensive (recommended on A10G/A100)."
    ),
)
@click.option(
    "--keep-temp",
    is_flag=True,
    default=False,
    help="Keep intermediate files (audio segments, etc.) after processing.",
)
@click.option(
    "--keep-bg",
    is_flag=True,
    default=False,
    help=(
        "Blend original background audio under the dub (legacy mode). "
        "Demucs separation is always used by default — this flag only applies "
        "when Demucs fails and is used as a fallback."
    ),
)
def main(
    url: str,
    output: str,
    model: str,
    voice: str,
    lang: str,
    clone_voice: bool,
    keep_temp: bool,
    keep_bg: bool,
):
    """
    \b
    Automated Video Dubbing System
    ─────────────────────────────────────
    Takes a YouTube URL (any language) and produces an English-dubbed video.
    Preserves original background music, SFX, and laughter via Demucs separation.

    URL is the YouTube video URL to dub (required).
    """
    console.print(BANNER, style="bold cyan")
    console.print(f"[dim]Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
    console.print(f"[bold]URL:[/bold] {url}")

    mode_label = "🎭 Coqui XTTS-v2 (Voice Cloning)" if clone_voice else "🗣  Edge-TTS (Fast, Two-Voice)"
    console.print(f"[bold]Mode:[/bold] {mode_label}")

    start_time = time.time()

    # ── Setup directories ────────────────────────────────────────────────────
    output_dir = Path(output)
    temp_dir   = output_dir / "temp"
    tts_dir    = temp_dir   / "tts_segments"

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    tts_dir.mkdir(parents=True, exist_ok=True)

    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output = str(output_dir / f"dubbed_{ts}.mp4")

    # ── Step 1: Download ─────────────────────────────────────────────────────
    total_steps = 6  # Download, Separate, Transcribe, Translate, Synthesize, Mix
    print_step(1, total_steps, "Downloading Video")
    from downloader import download_video
    video_path, audio_path = download_video(url, str(temp_dir))

    # ── Step 2: Vocal Separation (Demucs) ───────────────────────────────────
    print_step(2, total_steps, "Separating Vocals & Background (Demucs)")
    from vocal_separator import separate_vocals
    vocals_path, no_vocals_path = separate_vocals(audio_path, str(temp_dir / "demucs"))

    if no_vocals_path:
        console.print("[green]✓ Background preserved: music/SFX/laughing will appear in final dub[/green]")
        transcribe_source = vocals_path   # Whisper gets clean speech — better accuracy!
    else:
        console.print("[yellow]⚠  Demucs skipped — transcribing raw audio instead[/yellow]")
        transcribe_source = audio_path

    # ── Step 3: Transcribe ───────────────────────────────────────────────────
    print_step(3, total_steps, "Transcribing Audio")
    from transcriber import transcribe
    segments = transcribe(transcribe_source, model_size=model, language=lang)

    if not segments:
        console.print("[bold red]✗ No speech detected in the video. Exiting.[/bold red]")
        sys.exit(1)

    source_lang = segments[0].language

    # ── Step 4: Translate ────────────────────────────────────────────────────
    print_step(4, total_steps, "Translating to English")
    from translator import Translator
    translator = Translator(source_lang=source_lang)
    translated_segments = translator.translate_segments(segments)

    # ── Step 5: Synthesize ───────────────────────────────────────────────────
    print_step(5, total_steps, "Synthesizing English Speech")
    from synthesizer import synthesize_segments
    synthesized = synthesize_segments(
        translated_segments,
        str(tts_dir),
        voice_name=voice,
        clone_voice=clone_voice,
        vocals_path=vocals_path,
    )

    if not synthesized:
        console.print("[bold red]✗ TTS synthesis produced no output. Exiting.[/bold red]")
        sys.exit(1)

    # ── Step 6: Mix & Mux ───────────────────────────────────────────────────
    print_step(6, total_steps, "Mixing & Muxing Final Video")
    from audio_mixer import build_dubbed_audio, mux_audio_into_video

    dubbed_audio_path = build_dubbed_audio(
        synthesized,
        video_path,
        str(temp_dir),
        no_vocals_path=no_vocals_path,   # Demucs background (music/SFX/laughing)
        keep_background=keep_bg,          # Legacy fallback if Demucs failed
    )
    mux_audio_into_video(video_path, dubbed_audio_path, final_output)

    # ── Cleanup ──────────────────────────────────────────────────────────────
    if not keep_temp:
        console.print("[dim]🧹 Cleaning up temp files...[/dim]")
        shutil.rmtree(str(temp_dir), ignore_errors=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print_summary(
        url=url,
        video_path=video_path,
        output_path=final_output,
        source_lang=source_lang,
        num_segments=len(synthesized),
        elapsed=elapsed,
        mode=mode_label,
    )


if __name__ == "__main__":
    main()
