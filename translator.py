"""
translator.py
─────────────
Translates transcribed text segments into English.

Strategy (priority order):
1. IndicTrans2 (ai4bharat) — purpose-built for Indian languages (hi, pa, ur,
   mr, gu, ta, te, ml, kn, bn, etc.). Produces far more natural translations
   than generic models.  Uses GPU batching for speed.
2. Helsinki-NLP MarianMT — fast, offline fallback for all other languages.
3. Passthrough — if source is already English, or if all models fail.
"""

from dataclasses import dataclass
from rich.console import Console
from rich.progress import track

console = Console()

# ── IndicTrans2 language codes (Flores-200) ──────────────────────────────────
# Maps Whisper language codes → IndicTrans2 source language tags
INDIC_LANG_MAP: dict[str, str] = {
    "hi": "hin_Deva",   # Hindi
    "pa": "pan_Guru",   # Punjabi (Gurmukhi)
    "ur": "urd_Arab",   # Urdu
    "mr": "mar_Deva",   # Marathi
    "gu": "guj_Gujr",   # Gujarati
    "ta": "tam_Taml",   # Tamil
    "te": "tel_Telu",   # Telugu
    "ml": "mal_Mlym",   # Malayalam
    "kn": "kan_Knda",   # Kannada
    "bn": "ben_Beng",   # Bengali
    "or": "ory_Orya",   # Odia
    "as": "asm_Beng",   # Assamese
    "si": "sin_Sinh",   # Sinhala
    "ne": "npi_Deva",   # Nepali
}

# Target language for IndicTrans2
INDIC_TARGET = "eng_Latn"

# ── Helsinki-NLP fallback models ─────────────────────────────────────────────
HELSINKI_MODELS: dict[str, str] = {
    "de": "Helsinki-NLP/opus-mt-de-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
    "pt": "Helsinki-NLP/opus-mt-pt-en",
    "nl": "Helsinki-NLP/opus-mt-nl-en",
    "ru": "Helsinki-NLP/opus-mt-ru-en",
    "zh": "Helsinki-NLP/opus-mt-zh-en",
    "ja": "Helsinki-NLP/opus-mt-ja-en",
    "ko": "Helsinki-NLP/opus-mt-ko-en",
    "ar": "Helsinki-NLP/opus-mt-ar-en",
    "tr": "Helsinki-NLP/opus-mt-tr-en",
    "pl": "Helsinki-NLP/opus-mt-pl-en",
    "uk": "Helsinki-NLP/opus-mt-uk-en",
    "cs": "Helsinki-NLP/opus-mt-cs-en",
    "sv": "Helsinki-NLP/opus-mt-sv-en",
    "fi": "Helsinki-NLP/opus-mt-fi-en",
    "ro": "Helsinki-NLP/opus-mt-ro-en",
    "hu": "Helsinki-NLP/opus-mt-hu-en",
    "vi": "Helsinki-NLP/opus-mt-vi-en",
    "id": "Helsinki-NLP/opus-mt-id-en",
    "bg": "Helsinki-NLP/opus-mt-bg-en",
    "he": "Helsinki-NLP/opus-mt-he-en",
    "el": "Helsinki-NLP/opus-mt-el-en",
    "sk": "Helsinki-NLP/opus-mt-sk-en",
    "da": "Helsinki-NLP/opus-mt-da-en",
    "no": "Helsinki-NLP/opus-mt-no-en",
    # Fallback for unlisted Indian languages
    "hi": "Helsinki-NLP/opus-mt-hi-en",
    "bn": "Helsinki-NLP/opus-mt-bn-en",
}

ULTIMATE_FALLBACK = "Helsinki-NLP/opus-mt-hi-en"


class Translator:
    """
    Lazy-loading translator that picks the best available model for the language.

    Priority:
      1. IndicTrans2 for Indian languages (best quality for hi/pa/ur/mr/gu/ta/te/ml/kn/bn)
      2. Helsinki-NLP MarianMT for all other languages
      3. Passthrough if source is English or all models fail
    """

    def __init__(self, source_lang: str):
        self.source_lang = source_lang.lower()
        self._model_type = None   # 'indictrans2', 'marian', or 'passthrough'
        self._tokenizer  = None
        self._model      = None
        self._ip         = None   # IndicProcessor instance
        self._device     = "cpu"
        self._load_model()

    # ── Model loading ────────────────────────────────────────────────────────

    def _load_indictrans2(self) -> bool:
        """Try loading IndicTrans2. Returns True on success."""
        if self.source_lang not in INDIC_LANG_MAP:
            return False
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            # ── Compatibility patch ───────────────────────────────────────────
            # IndicTransToolkit imports PreTrainedTokenizerBase from the old
            # location (transformers.tokenization_utils). In transformers ≥4.44
            # it was moved to tokenization_utils_base. Patch before importing.
            import transformers.tokenization_utils as _tu
            if not hasattr(_tu, "PreTrainedTokenizerBase"):
                from transformers.tokenization_utils_base import PreTrainedTokenizerBase
                _tu.PreTrainedTokenizerBase = PreTrainedTokenizerBase
            # ─────────────────────────────────────────────────────────────────

            from IndicTransToolkit.processor import IndicProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model_name = "ai4bharat/indictrans2-indic-en-1B"
            console.print(f"[bold cyan]🌐 Loading IndicTrans2 ({model_name}) on {self._device}...[/bold cyan]")

            self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
            ).to(self._device)
            self._model.eval()

            self._ip = IndicProcessor(inference=True)
            self._model_type = "indictrans2"
            console.print(f"[green]✓ IndicTrans2 loaded on {self._device}[/green]")
            return True

        except Exception as e:
            console.print(f"[yellow]⚠  IndicTrans2 unavailable: {e}[/yellow]")
            console.print("[yellow]   Falling back to MarianMT. For better Hindi/Indic quality,[/yellow]")
            console.print("[yellow]   run: pip install IndicTransToolkit[/yellow]")
            return False

    def _load_marian(self, model_name: str) -> bool:
        """Load a MarianMT model. Returns True on success."""
        try:
            import torch
            from transformers import MarianMTModel, MarianTokenizer

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            console.print(f"[bold cyan]🌐 Loading MarianMT ({model_name}) on {self._device}...[/bold cyan]")

            self._tokenizer = MarianTokenizer.from_pretrained(model_name)
            self._model = MarianMTModel.from_pretrained(model_name).to(self._device)
            self._model.eval()
            self._model_type = "marian"
            console.print(f"[green]✓ Loaded MarianMT: {model_name} on {self._device}[/green]")
            return True
        except Exception as e:
            console.print(f"[yellow]⚠  Failed to load {model_name}: {e}[/yellow]")
            return False

    def _load_model(self):
        """Load the best available translation model for the source language."""
        lang = self.source_lang

        if lang == "en":
            console.print("[yellow]⚠  Source language is already English, skipping translation.[/yellow]")
            self._model_type = "passthrough"
            return

        # 1. Try IndicTrans2 for Indian languages
        if lang in INDIC_LANG_MAP:
            if self._load_indictrans2():
                return

        # 2. Try Helsinki MarianMT for the specific language
        if lang in HELSINKI_MODELS:
            if self._load_marian(HELSINKI_MODELS[lang]):
                return

        # 3. Ultimate fallback (hi-en covers most Indo-Aryan langs)
        console.print(f"[yellow]⚠  No model for '{lang}', falling back to {ULTIMATE_FALLBACK}[/yellow]")
        if self._load_marian(ULTIMATE_FALLBACK):
            return

        console.print("[red]✗ All translation models failed — output will be in original language[/red]")
        self._model_type = "passthrough"

    # ── Translation ──────────────────────────────────────────────────────────

    def _translate_batch_indictrans2(self, texts: list[str]) -> list[str]:
        """Batch-translate using IndicTrans2."""
        import torch

        src_lang = INDIC_LANG_MAP[self.source_lang]
        batch = self._ip.preprocess_batch(texts, src_lang=src_lang, tgt_lang=INDIC_TARGET)

        inputs = self._tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self._device)

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                num_beams=5,
                num_return_sequences=1,
                max_length=256,
            )

        decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
        return self._ip.postprocess_batch(decoded, lang=INDIC_TARGET)

    def _translate_batch_marian(self, texts: list[str]) -> list[str]:
        """Batch-translate using MarianMT."""
        import torch

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self._device)

        with torch.no_grad():
            generated = self._model.generate(**inputs, max_length=512)

        return [self._tokenizer.decode(g, skip_special_tokens=True) for g in generated]

    def translate_text(self, text: str) -> str:
        """Translate a single string to English."""
        if not text.strip() or self._model_type == "passthrough":
            return text
        try:
            results = (
                self._translate_batch_indictrans2([text])
                if self._model_type == "indictrans2"
                else self._translate_batch_marian([text])
            )
            return results[0] if results else text
        except Exception as e:
            console.print(f"[red]⚠  Translation error: {e}[/red]")
            return text

    def translate_segments(self, segments: list, batch_size: int = 32) -> list:
        """
        Translate all segments in GPU-accelerated batches.

        Batching is far faster than translating one segment at a time:
        - IndicTrans2: ~32 segments per GPU call vs. 1 at a time
        - MarianMT:    same benefit

        Args:
            segments:   List of Segment objects (from transcriber.py).
            batch_size: Number of segments per translation batch.

        Returns:
            Same list with .text updated to English translation.
        """
        if self._model_type == "passthrough":
            console.print("[green]✓ No translation needed (already English)[/green]")
            return segments

        console.print(f"[bold cyan]🌐 Translating {len(segments)} segments to English "
                      f"(batch_size={batch_size}, device={self._device})...[/bold cyan]")

        from transcriber import Segment

        translated = []
        batches = [segments[i:i + batch_size] for i in range(0, len(segments), batch_size)]

        for batch in track(batches, description="Translating batches...", console=console):
            texts = [seg.text for seg in batch]
            try:
                if self._model_type == "indictrans2":
                    translated_texts = self._translate_batch_indictrans2(texts)
                else:
                    translated_texts = self._translate_batch_marian(texts)
            except Exception as e:
                console.print(f"[red]⚠  Batch translation error: {e}[/red]")
                translated_texts = texts  # passthrough on error

            # Pad result list in case of partial batch failures
            while len(translated_texts) < len(batch):
                translated_texts.append(batch[len(translated_texts)].text)

            for seg, eng_text in zip(batch, translated_texts):
                translated.append(Segment(
                    start=seg.start,
                    end=seg.end,
                    text=eng_text,
                    language="en",
                ))

        console.print(f"[green]✓ Translation complete: {len(translated)} segments[/green]")
        return translated
