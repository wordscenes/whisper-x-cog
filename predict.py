import json
import os
import typing
from importlib.metadata import version

from cog import BasePredictor, Path, Input
import whisperx


# all model files will be downloaded to this directory
MODEL_CACHE = Path(__file__).parent / 'model_cache'
temperature_increment_on_fallback = 0.2

# Not v3! See "rejected experiments" in readme
WHISPER_MODEL = 'large-v2'

TARGET_LANGUAGES = [
    'ar',
    'zh',
    'en',
    'fr',
    'de',
    'hi',
    'id',
    'it',
    'ja',
    'ko',
    'pt',
    'ru',
    'es',
    'tr',
]

OUR_ALIGN_MODELS_HF = {
    'id': 'indonesian-nlp/wav2vec2-large-xlsr-indonesian',
}

def report_versions():
    print(f'Using whisperx version {version("whisperx")}')
    print(f'Using Whisper model {WHISPER_MODEL}')


class Predictor(BasePredictor):
    def setup(self, weights = None):
        """Load the model into memory to make running multiple predictions efficient"""

        # Ensure the model cache directory exists
        os.makedirs(MODEL_CACHE, exist_ok=True)

        self.model = whisperx.load_model(
            WHISPER_MODEL,
            device='cuda',
            compute_type='float16',
            download_root=str(MODEL_CACHE),
        )
        self.align_models = {}
        for lang in TARGET_LANGUAGES:
            self.align_models[lang] = whisperx.load_align_model(
                language_code=lang,
                device='cuda',
                model_name = OUR_ALIGN_MODELS_HF.get(lang, None),
                model_dir=str(MODEL_CACHE),
            )

    def predict(  # pyright: ignore[reportIncompatibleMethodOverride] (Pyright complains about **kwargs missing, but cog doesn't support untyped parameters)
        self,
        audio_path: Path = Input(description='Audio to transcribe or align'),
        mode: str = Input(default="transcribe", choices=["transcribe", "align"], description="Mode: 'transcribe' to generate transcript, 'align' to align provided segments"),
        segments: str = Input(default="", description="Segments (JSON array with text, start, and end keys) to align with audio (required when mode='align')"),
        language: str = Input(default="en", description="Language to transcribe"),
    ) -> str:
        report_versions()

        audio = whisperx.load_audio(audio_path)
        
        if mode == "transcribe":
            result = self.model.transcribe(audio, language=language)
        else:
            result = {'segments': json.loads(segments)}

        align_model = self.align_models[language]
        result = whisperx.align(
            result['segments'],
            align_model[0],
            align_model[1],
            audio,
            'cuda',
            return_char_alignments=False,
        )

        if not isinstance(result, dict) and callable(getattr(result, 'to_dict')):
            result = result.to_dict()
        output = json.dumps(result, allow_nan=True, ensure_ascii=False)

        return output
