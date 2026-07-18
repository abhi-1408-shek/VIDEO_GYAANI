"""
vocal_separator.py
──────────────────
Separates an audio file into vocals and background (music/SFX/laughing)
using Facebook's Demucs neural network.

Why Demucs?
- Purpose-built deep learning model for music source separation.
- The 2-stem mode (vocals vs. everything else) is fast and extremely clean.
- Running it means the final dub has ZERO original-language bleed-through,
  while keeping 100% of the background music, laughter, and sound effects.

Output:
- vocals.wav      : Only the human speech (used for Whisper + XTTS reference)
- no_vocals.wav   : Background music, SFX, laughing — everything except speech
                    (used as the base canvas for the final dubbed audio track)
"""

import glob
import subprocess
from pathlib import Path

import torch
from rich.console import Console

console = Console()


def separate_vocals(
    audio_path: str,
    output_dir: str,
) -> tuple[str, str | None]:
    """
    Run Demucs (htdemucs, 2-stem mode) to split audio into vocals and background.

    Args:
        audio_path:  Path to the full-quality source audio (44100Hz stereo WAV).
        output_dir:  Directory to write Demucs outputs.

    Returns:
        (vocals_path, no_vocals_path)
        If Demucs fails for any reason, returns (audio_path, None) as a safe fallback
        so the rest of the pipeline can continue without crashing.
    """
    audio_path = str(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print(
        f"[bold cyan]🎵 Separating vocals from background using Demucs "
        f"(device={device})...[/bold cyan]"
    )
    console.print("[dim]  This separates speech from music/SFX/laughing for a clean dub[/dim]")

    # Demucs CLI: --two-stems vocals → produce vocals.wav + no_vocals.wav
    # Model htdemucs is the default hybrid transformer model (best quality)
    cmd = [
        "python", "-m", "demucs",
        "--two-stems", "vocals",
        "--out", str(output_dir),
        "--device", device,
        audio_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        console.print(f"[yellow]⚠  Demucs failed (return code {result.returncode})[/yellow]")
        console.print(f"[dim]{result.stderr[-500:]}[/dim]")
        console.print("[yellow]   Falling back to raw audio (no vocal separation)[/yellow]")
        return audio_path, None

    # Demucs creates: <output_dir>/<model_name>/<audio_stem>/vocals.wav
    # Model name is 'htdemucs' (the default), audio_stem = Path(audio_path).stem
    audio_stem = Path(audio_path).stem

    # Use glob to be robust to model name variations
    vocals_candidates = glob.glob(
        str(output_dir / "*" / audio_stem / "vocals.wav")
    )
    no_vocals_candidates = glob.glob(
        str(output_dir / "*" / audio_stem / "no_vocals.wav")
    )

    if not vocals_candidates or not no_vocals_candidates:
        console.print("[yellow]⚠  Demucs output not found — falling back to raw audio[/yellow]")
        console.print(f"[dim]Searched in: {output_dir}/*/{audio_stem}/vocals.wav[/dim]")
        return audio_path, None

    vocals_path    = vocals_candidates[0]
    no_vocals_path = no_vocals_candidates[0]

    console.print(f"[green]✓ Vocals (speech only):[/green] {vocals_path}")
    console.print(f"[green]✓ Background (BGM/SFX/laughing):[/green] {no_vocals_path}")
    return vocals_path, no_vocals_path
