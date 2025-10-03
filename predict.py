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

def report_versions():
    print(f'Using whisperx version {version("whisperx")}')
    print('*Not* using faster-whisper')  # see "rejected experiments" in readme
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
                download_root=str(MODEL_CACHE),
            )

    def predict(self) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        return "Hello, world!"
