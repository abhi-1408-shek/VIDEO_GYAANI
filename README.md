# Automated Video Dubbing System

A Python pipeline that takes any YouTube video (in any language) and produces an English-dubbed version with natural-sounding speech, precisely synchronized to the original video.

---

## What It Does

1. **Downloads** the YouTube video using `yt-dlp`
2. **Separates** vocals from background using `Demucs` (music, SFX, and laughing are preserved perfectly)
3. **Transcribes** clean speech with timestamps using `faster-whisper` (Whisper large-v2, GPU)
4. **Translates** to English using **IndicTrans2** (Indian languages) or Helsinki-NLP MarianMT (others)
5. **Synthesizes** English speech — two modes:
   - **Fast (default):** Edge-TTS with two-voice speaker detection and adaptive rate control
   - **Voice-cloning (`--clone-voice`):** Coqui XTTS-v2 using per-segment Demucs reference clips
6. **Mixes** TTS audio over the Demucs background track (numpy overlay, perfect sync)
7. **Muxes** the final dubbed audio into the original video without re-encoding

---

## Examples

Below are examples demonstrating the pipeline's voice cloning and background audio preservation capabilities:

- **Original Video:** [Link to Original Video]
- **Dubbed Output:** [Link to Dubbed Video]

*(Upload your original and dubbed output videos here to showcase the system's quality.)*

---

## Quick Start (Google Colab)

Paste this into a Colab cell and run it once per session:

```python
# 1. Upload video_dubber_clean.zip, then:
import zipfile, os
with zipfile.ZipFile("video_dubber_clean.zip", "r") as z:
    z.extractall(".")
os.chdir("/content/video_dubber")

# 2. System dependencies
!apt-get install -y ffmpeg > /dev/null 2>&1

# 3. Core Python packages
!pip install -q yt-dlp faster-whisper edge-tts pydub transformers \
    torch sentencepiece sacremoses accelerate rich click numpy soundfile

# 4. Demucs (vocal/BGM separation — keeps laughter, music, SFX)
!pip install -q demucs

# 5. Coqui TTS (voice cloning — only needed for --clone-voice flag)
!pip install -q TTS

# 6. IndicTransToolkit (best Hindi/Indic translation)
!pip install -q git+https://github.com/VarunGumma/IndicTransToolkit

# 7a. Fast mode — Edge-TTS voices, ~8 min for 30-min video on T4
!python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --model large-v2

# 7b. Voice-cloning mode — Coqui XTTS-v2, best on A10G/A100
!python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --model large-v2 --clone-voice
```

---

## Local Setup

### Prerequisites

- Python 3.10+
- `ffmpeg` installed and in your PATH
  - **macOS**: `brew install ffmpeg`
  - **Ubuntu**: `sudo apt install ffmpeg`

### Setup

```bash
cd video_dubber

# Run the one-shot setup script
chmod +x setup.sh
./setup.sh

# Activate the environment
source .venv/bin/activate
```

### Run

```bash
# Basic usage (30-min video, auto language detection)
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --model large-v2

# Choose a voice
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --voice female --model large-v2

# Keep original background audio blended at low volume
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --keep-bg

# Specify output directory
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --output ./my_dubs
```

---

## CLI Options

| Option | Default | Description |
|---|---|---|
| `URL` | (required) | YouTube URL to dub |
| `--output, -o` | `./output` | Output directory |
| `--model, -m` | `medium` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` |
| `--voice, -v` | `default` | TTS voice: `default`, `male`, `female`, `male_alt`, `female_alt`, `neutral` |
| `--keep-temp` | `False` | Keep intermediate files after processing |
| `--keep-bg` | `False` | Blend original audio at low volume under the dubbed speech |
| `--bg-volume` | `-5.0` | Volume of background track (claps/laughs/music) in decibels |
| `--reset` | `False` | Clear cached files and restart processing from scratch |
| `--clone-voice` | `False` | Use Coqui XTTS-v2 for zero-shot voice cloning |

---

## Architecture

```
main.py           ← CLI entry point (orchestrates the pipeline)
│
├── downloader.py     ← Downloads video (yt-dlp) & extracts audio (ffmpeg)
├── transcriber.py    ← Transcribes audio into timestamped segments (faster-whisper)
├── translator.py     ← Translates segments to English (IndicTrans2 → MarianMT fallback)
├── synthesizer.py    ← Synthesizes English TTS per segment (edge-tts, async batch)
└── audio_mixer.py    ← Numpy-buffer mixing for perfect sync + ffmpeg atempo stretch
```

### Key Design Decisions

- **Numpy buffer mixing**: The entire video duration is allocated as a numpy float32 array. Each TTS segment is written at its **exact original millisecond timestamp** — mathematically perfect sync regardless of TTS length.
- **ffmpeg atempo time-stretching**: If a TTS segment is longer than its original time slot, it is time-stretched using ffmpeg's `atempo` filter (much higher quality than pydub's `speedup`). Capped at 2x speed.
- **Hard trim at next segment boundary**: If a stretched segment would overflow into the next speaker's slot, it is hard-trimmed. Better to cut a word than drift out of sync for the rest of the video.
- **IndicTrans2 for Indian languages**: Specifically designed for Hindi, Punjabi, Marathi, Gujarati, Tamil, Telugu, etc. Produces far more natural English translations than generic models.
- **Adaptive TTS rate**: Before TTS synthesis, we estimate how much a segment will need to be compressed. If >1.3x compression is needed, we request a faster speaking rate from Edge TTS, reducing the load on post-processing.
- **Two-voice speaker detection**: Uses silence gaps (>2.5s) as a heuristic for speaker changes and alternates between a male and female voice — no diarization library needed.
- **Stream copy**: `ffmpeg -c:v copy` avoids re-encoding the video — keeps quality intact and is very fast.
- **Language auto-detection**: Whisper detects the source language automatically; the translator picks the best model for that language.

---

## Supported Languages

Any language supported by Whisper for transcription (~100 languages). For translation:

- **IndicTrans2** (primary for Indian langs): Hindi, Punjabi, Urdu, Marathi, Gujarati, Tamil, Telugu, Malayalam, Kannada, Bengali, Odia, Assamese, Nepali
- **Helsinki-NLP MarianMT** (all others): German, French, Spanish, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, and 20+ more

---

## Available Voices

| Key | Voice | Description |
|---|---|---|
| `default` | `en-US-AndrewMultilingualNeural` | Natural male (recommended) |
| `male` | `en-US-AndrewMultilingualNeural` | Natural male |
| `female` | `en-US-AvaMultilingualNeural` | Natural female |
| `male_alt` | `en-GB-RyanNeural` | British male |
| `female_alt` | `en-GB-SoniaNeural` | British female |
| `neutral` | `en-US-EmmaMultilingualNeural` | Neutral US female |

> **Two-voice mode**: The pipeline automatically detects speaker changes using silence gaps and alternates between male and female voices — no diarization needed.

---

## Output Structure

```
output/
├── dubbed_20240718_120000.mp4    ← Final dubbed video
└── temp/                         ← Intermediate files (deleted unless --keep-temp)
    ├── source_video.mp4
    ├── source_audio.wav
    ├── dubbed_audio.wav
    └── tts_segments/
        ├── seg_00000.mp3
        ├── seg_00001.mp3
        └── ...
```

---

## Performance (Google Colab T4 GPU)

| Step | Tool | Time (30-min video) | Time (2-hour video) |
|---|---|---|---|
| Download | yt-dlp | ~15 sec | ~60 sec |
| Transcription | Whisper large-v2 (GPU) | ~4 min | ~15 min |
| Translation | IndicTrans2 (GPU, batched) | ~1 min | ~4 min |
| TTS Synthesis | edge-tts (15 concurrent) | ~2 min | ~8 min |
| Audio Mixing | numpy + ffmpeg atempo | ~10 sec | ~30 sec |
| **Total** | | **~7–8 min** | **~28–30 min** |

---

## Dependencies

| Package | Purpose |
|---|---|
| `yt-dlp` | YouTube video downloading |
| `faster-whisper` | Speech transcription with timestamps |
| `IndicTransToolkit` | IndicTrans2 for Indian language translation |
| `transformers` | MarianMT fallback translation models |
| `edge-tts` | Natural TTS via Microsoft Edge |
| `pydub` | Audio loading and format conversion |
| `numpy` | Fast numpy-buffer audio mixing |
| `soundfile` | Audio I/O via libsndfile |
| `ffmpeg` | atempo time-stretching + video muxing |
| `rich` | Beautiful terminal output |
| `click` | CLI argument parsing |
